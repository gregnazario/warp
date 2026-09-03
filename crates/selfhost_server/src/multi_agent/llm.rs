use anyhow::{Context as _, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use futures::StreamExt as _;
use serde_json::{Value, json};

use crate::config::{LlmSchema, ResolvedLlm};

/// A tool definition advertised to the model.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON schema for the tool's arguments.
    pub parameters: Value,
}

/// A single assistant tool call as emitted by the model.
#[derive(Debug, Clone)]
pub struct ToolCallOut {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// An image attached to a user message, as raw bytes plus its mime type.
#[derive(Debug, Clone)]
pub struct LlmImage {
    pub data: Vec<u8>,
    pub mime_type: String,
}

/// A provider-neutral LLM conversation message.
#[derive(Debug, Clone)]
pub enum LlmMessage {
    User {
        text: String,
        images: Vec<LlmImage>,
    },
    Assistant {
        text: String,
        tool_calls: Vec<ToolCallOut>,
    },
    ToolResult {
        tool_call_id: String,
        text: String,
    },
}

/// Incremental output from a streaming LLM completion.
///
/// Tool calls arrive as a delta that opens the call (with an id or name) or
/// appends an argument fragment to the most recently opened call.
#[derive(Debug, Clone)]
pub enum Delta {
    Text(String),
    ToolCall {
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
}

/// Token usage reported by the provider for one completion.
#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Streams one LLM completion, invoking `on_delta` for each incremental piece.
///
/// Returns the usage reported by the provider.
pub async fn stream_completion(
    client: &reqwest::Client,
    llm: &ResolvedLlm,
    system: &str,
    messages: &[LlmMessage],
    tools: &[ToolDef],
    on_delta: impl FnMut(Delta) -> Result<()>,
) -> Result<Usage> {
    match llm.schema {
        LlmSchema::Openai => openai::stream(client, llm, system, messages, tools, on_delta).await,
        LlmSchema::Anthropic => {
            anthropic::stream(client, llm, system, messages, tools, on_delta).await
        }
        LlmSchema::Chatgpt => chatgpt::stream(client, llm, system, messages, tools, on_delta).await,
    }
}

async fn post_stream(
    client: &reqwest::Client,
    url: String,
    headers: Vec<(String, String)>,
    body: Value,
) -> Result<reqwest::Response> {
    let mut request = client.post(url).json(&body);
    for (name, value) in headers {
        request = request.header(name.as_str(), value);
    }
    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        bail!("LLM endpoint returned {status}: {}", truncate(&body, 2000));
    }
    Ok(response)
}

/// Parses an SSE body of `data:` lines, invoking `on_data` per payload.
async fn for_each_sse_data(
    response: reqwest::Response,
    mut on_data: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        buffer.extend_from_slice(&chunk?);
        while let Some(pos) = buffer.windows(2).position(|window| window == b"\n\n") {
            let block: Vec<u8> = buffer.drain(..pos + 2).collect();
            for line in String::from_utf8_lossy(&block).lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    on_data(data.trim())?;
                }
            }
        }
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

fn tool_call_output(tool_calls: &[ToolCallOut]) -> Vec<Value> {
    tool_calls
        .iter()
        .map(|call| {
            json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments.to_string(),
                },
            })
        })
        .collect()
}

fn tool_definitions(tools: &[ToolDef]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        })
        .collect()
}

mod openai {
    use super::*;

    pub(super) async fn stream(
        client: &reqwest::Client,
        llm: &ResolvedLlm,
        system: &str,
        messages: &[LlmMessage],
        tools: &[ToolDef],
        mut on_delta: impl FnMut(Delta) -> Result<()>,
    ) -> Result<Usage> {
        let mut payload_messages = vec![json!({"role": "system", "content": system})];
        for message in messages {
            match message {
                LlmMessage::User { text, images } => {
                    if images.is_empty() {
                        payload_messages.push(json!({"role": "user", "content": text}));
                    } else {
                        let mut content = vec![json!({"type": "text", "text": text})];
                        for image in images {
                            content.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!(
                                        "data:{};base64,{}",
                                        image.mime_type,
                                        BASE64_STANDARD.encode(&image.data)
                                    ),
                                },
                            }));
                        }
                        payload_messages.push(json!({"role": "user", "content": content}));
                    }
                }
                LlmMessage::Assistant { text, tool_calls } => {
                    let mut entry = json!({"role": "assistant"});
                    if !text.is_empty() {
                        entry["content"] = Value::String(text.clone());
                    }
                    if !tool_calls.is_empty() {
                        entry["tool_calls"] = Value::Array(tool_call_output(tool_calls));
                    }
                    payload_messages.push(entry);
                }
                LlmMessage::ToolResult { tool_call_id, text } => {
                    payload_messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": text,
                    }));
                }
            }
        }

        let mut body = json!({
            "model": llm.model,
            "messages": payload_messages,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tool_definitions(tools));
        }

        let url = format!("{}/chat/completions", llm.base_url.trim_end_matches('/'));
        let mut headers = Vec::new();
        if let Some(key) = &llm.api_key {
            headers.push(("Authorization".to_owned(), format!("Bearer {key}")));
        }
        headers.extend(llm.headers.clone());
        let response = post_stream(client, url, headers, body).await?;

        let mut usage = Usage::default();
        // (id, name, argument fragments) per tool call, keyed by the
        // provider's call index.
        let mut open_calls: std::collections::BTreeMap<usize, (String, String, String)> =
            std::collections::BTreeMap::new();
        for_each_sse_data(response, |data| {
            if data == "[DONE]" {
                return Ok(());
            }
            let chunk: Value = serde_json::from_str(data).with_context(|| {
                format!("failed to decode OpenAI chunk: {}", truncate(data, 400))
            })?;
            if let Some(u) = chunk.get("usage") {
                usage.input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0);
                usage.output_tokens = u["completion_tokens"].as_u64().unwrap_or(0);
            }
            let Some(delta) = chunk["choices"][0]["delta"].as_object() else {
                return Ok(());
            };
            if let Some(text) = delta.get("content").and_then(Value::as_str)
                && !text.is_empty()
            {
                on_delta(Delta::Text(text.to_owned()))?;
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in tool_calls {
                    let index = call["index"].as_u64().unwrap_or(0) as usize;
                    let entry = open_calls.entry(index).or_default();
                    if let Some(id) = call["id"].as_str() {
                        entry.0.push_str(id);
                    }
                    if let Some(name) = call["function"]["name"].as_str() {
                        entry.1.push_str(name);
                    }
                    // The spec streams arguments as string fragments, but some
                    // OpenAI-compatible endpoints (e.g. Ollama) send the
                    // arguments as a complete JSON object.
                    if let Some(arguments) = call["function"]["arguments"].as_str() {
                        entry.2.push_str(arguments);
                    } else if !call["function"]["arguments"].is_null() {
                        entry.2.push_str(&call["function"]["arguments"].to_string());
                    }
                }
            }
            Ok(())
        })
        .await
        .context("OpenAI SSE stream failed")?;

        // OpenAI streams tool calls as fragments; flush them as whole calls
        // once the stream is done.
        for (id, name, arguments) in open_calls.into_values() {
            if name.is_empty() {
                continue;
            }
            on_delta(Delta::ToolCall {
                id: (!id.is_empty()).then_some(id),
                name: Some(name),
                arguments,
            })?;
        }
        Ok(usage)
    }
}

mod anthropic {
    use super::*;

    pub(super) async fn stream(
        client: &reqwest::Client,
        llm: &ResolvedLlm,
        system: &str,
        messages: &[LlmMessage],
        tools: &[ToolDef],
        mut on_delta: impl FnMut(Delta) -> Result<()>,
    ) -> Result<Usage> {
        let mut payload_messages = Vec::new();
        for message in messages {
            match message {
                LlmMessage::User { text, images } => {
                    let mut content: Vec<Value> = images
                        .iter()
                        .map(|image| {
                            json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image.mime_type,
                                    "data": BASE64_STANDARD.encode(&image.data),
                                },
                            })
                        })
                        .collect();
                    content.push(json!({"type": "text", "text": text}));
                    payload_messages.push(json!({"role": "user", "content": content}));
                }
                LlmMessage::Assistant { text, tool_calls } => {
                    let mut content = Vec::new();
                    if !text.is_empty() {
                        content.push(json!({"type": "text", "text": text}));
                    }
                    for call in tool_calls {
                        content.push(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }));
                    }
                    payload_messages.push(json!({"role": "assistant", "content": content}));
                }
                LlmMessage::ToolResult { tool_call_id, text } => {
                    payload_messages.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": [{"type": "text", "text": text}],
                        }],
                    }));
                }
            }
        }

        let mut body = json!({
            "model": llm.model,
            "max_tokens": 16_384,
            // Mark the system prompt as cacheable so the stable prefix
            // (system prompt + tool definitions) is prompt-cached.
            "system": [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": payload_messages,
            "stream": true,
        });
        if !tools.is_empty() {
            let definitions: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    })
                })
                .collect();
            body["tools"] = Value::Array(definitions);
        }

        let url = format!("{}/messages", llm.base_url.trim_end_matches('/'));
        let mut headers = vec![("anthropic-version".to_owned(), "2023-06-01".to_owned())];
        if let Some(key) = &llm.api_key {
            headers.push(("x-api-key".to_owned(), key.clone()));
        }
        headers.extend(llm.headers.clone());
        let response = post_stream(client, url, headers, body).await?;

        let mut usage = Usage::default();
        for_each_sse_data(response, |data| {
            let event: Value = serde_json::from_str(data).with_context(|| {
                format!("failed to decode Anthropic event: {}", truncate(data, 400))
            })?;
            match event["type"].as_str() {
                Some("message_start") => {
                    usage.input_tokens = event["message"]["usage"]["input_tokens"]
                        .as_u64()
                        .unwrap_or(0);
                }
                Some("content_block_start") => {
                    let block = &event["content_block"];
                    if block["type"] == "tool_use" {
                        on_delta(Delta::ToolCall {
                            id: block["id"].as_str().map(ToOwned::to_owned),
                            name: block["name"].as_str().map(ToOwned::to_owned),
                            arguments: String::new(),
                        })?;
                    }
                }
                Some("content_block_delta") => {
                    let delta = &event["delta"];
                    match delta["type"].as_str() {
                        Some("text_delta") => {
                            if let Some(text) = delta["text"].as_str()
                                && !text.is_empty()
                            {
                                on_delta(Delta::Text(text.to_owned()))?;
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(fragment) = delta["partial_json"].as_str()
                                && !fragment.is_empty()
                            {
                                on_delta(Delta::ToolCall {
                                    id: None,
                                    name: None,
                                    arguments: fragment.to_owned(),
                                })?;
                            }
                        }
                        _ => {}
                    }
                }
                Some("message_delta") => {
                    usage.output_tokens = event["usage"]["output_tokens"].as_u64().unwrap_or(0);
                }
                _ => {}
            }
            Ok(())
        })
        .await
        .context("Anthropic SSE stream failed")?;
        Ok(usage)
    }
}

mod chatgpt {
    use super::*;

    /// Streams one completion from the ChatGPT-backend Codex endpoint, which
    /// speaks the OpenAI Responses API instead of chat completions.
    pub(super) async fn stream(
        client: &reqwest::Client,
        llm: &ResolvedLlm,
        system: &str,
        messages: &[LlmMessage],
        tools: &[ToolDef],
        mut on_delta: impl FnMut(Delta) -> Result<()>,
    ) -> Result<Usage> {
        let mut input = Vec::new();
        for message in messages {
            match message {
                LlmMessage::User { text, images } => {
                    let mut content: Vec<Value> = images
                        .iter()
                        .map(|image| {
                            json!({
                                "type": "input_image",
                                "image_url": format!(
                                    "data:{};base64,{}",
                                    image.mime_type,
                                    BASE64_STANDARD.encode(&image.data)
                                ),
                            })
                        })
                        .collect();
                    content.push(json!({"type": "input_text", "text": text}));
                    input.push(json!({"type": "message", "role": "user", "content": content}));
                }
                LlmMessage::Assistant { text, tool_calls } => {
                    if !text.is_empty() {
                        input.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }));
                    }
                    for call in tool_calls {
                        input.push(json!({
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": call.arguments.to_string(),
                        }));
                    }
                }
                LlmMessage::ToolResult { tool_call_id, text } => {
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": text,
                    }));
                }
            }
        }

        let tool_definitions: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": false,
                })
            })
            .collect();

        let body = json!({
            "model": llm.model,
            "instructions": system,
            "input": input,
            "tools": tool_definitions,
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "store": false,
            "stream": true,
        });

        let url = format!("{}/responses", llm.base_url.trim_end_matches('/'));
        let mut request = client
            .post(url)
            .header(
                "Authorization",
                format!("Bearer {}", llm.api_key.clone().unwrap_or_default()),
            )
            .header("OpenAI-Beta", "responses=experimental")
            .header("Accept", "text/event-stream")
            .header("originator", "selfhost-server")
            .json(&body);
        if let Some(account_id) = &llm.account_id {
            request = request.header("chatgpt-account-id", account_id);
        }
        let response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!(
                "Codex endpoint returned {status}: {}",
                truncate(&body, 2000)
            );
        }

        let mut usage = Usage::default();
        for_each_sse_data(response, |data| {
            let event: Value = serde_json::from_str(data).with_context(|| {
                format!("failed to decode Codex event: {}", truncate(data, 400))
            })?;
            match event["type"].as_str() {
                Some("response.output_text.delta") => {
                    if let Some(text) = event["delta"].as_str()
                        && !text.is_empty()
                    {
                        on_delta(Delta::Text(text.to_owned()))?;
                    }
                }
                Some("response.output_item.added") => {
                    let item = &event["item"];
                    if item["type"] == "function_call" {
                        on_delta(Delta::ToolCall {
                            id: item["call_id"].as_str().map(ToOwned::to_owned),
                            name: item["name"].as_str().map(ToOwned::to_owned),
                            arguments: String::new(),
                        })?;
                    }
                }
                Some("response.function_call_arguments.delta") => {
                    if let Some(fragment) = event["delta"].as_str()
                        && !fragment.is_empty()
                    {
                        on_delta(Delta::ToolCall {
                            id: None,
                            name: None,
                            arguments: fragment.to_owned(),
                        })?;
                    }
                }
                Some("response.completed" | "response.incomplete") => {
                    usage.input_tokens = event["response"]["usage"]["input_tokens"]
                        .as_u64()
                        .unwrap_or(0);
                    usage.output_tokens = event["response"]["usage"]["output_tokens"]
                        .as_u64()
                        .unwrap_or(0);
                }
                Some("response.failed") => {
                    let message = event["response"]["error"]["message"]
                        .as_str()
                        .unwrap_or("unknown Codex failure");
                    bail!("Codex request failed: {message}");
                }
                _ => {}
            }
            Ok(())
        })
        .await
        .context("Codex SSE stream failed")?;
        Ok(usage)
    }
}

#[cfg(test)]
#[path = "llm_tests.rs"]
mod tests;
