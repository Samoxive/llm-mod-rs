use anyhow::Result;
use async_trait::async_trait;
use mistralrs::Constraint::JsonSchema;
use mistralrs::{Model, RequestBuilder, TextMessageRole, TextModelBuilder};
use serde::Deserialize;
use serde_json::json;
use serenity::all::{
    ChannelId, Context, CreateEmbed, CreateMessage, EventHandler, GatewayIntents, Message,
};
use serenity::Client;
use std::env;
use tracing::error;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Deserialize, Debug)]
struct Evaluation {
    violates_rules: bool,
}

async fn evaluate_message(model: &Model, content: &str) -> bool {
    let request = RequestBuilder::new()
        .set_sampler_temperature(0.1)
        .set_sampler_max_len(100)
        .set_constraint(JsonSchema(json!(
            {
                "type": "object",
                "properties": {
                    "violates_rules": {"type": "boolean"},
                },
                "required": ["violates_rules"],
                "additionalProperties": false,
            }
        )))
        .add_message(
            TextMessageRole::System,
            "You are a moderator for a Discord community.\
            In this community users aren't allowed send messages containing job posts.\
            Users can't list their skills to attract recruiters.\
            You will be provided messages to evaluate whether user breaks these rules and you will respond true if it breaks rules, false if it doesn't.",
        )
        .add_message(
            TextMessageRole::User,
            content,
        );

    let response = match model.send_chat_request(request).await {
        Ok(response) => response,
        Err(err) => {
            error!("failed to run llm {:?}", err);
            return false;
        }
    };

    let choice = match response.choices.first() {
        Some(choice) => choice,
        None => {
            error!("llm returned zero choices");
            return false;
        }
    };

    let response_content = match choice.message.content {
        Some(ref content) => content,
        None => {
            error!("llm returned choice without content");
            return false;
        }
    };

    let evaluation = match serde_json::from_str::<Evaluation>(response_content) {
        Ok(evaluation) => evaluation,
        Err(err) => {
            error!(
                "failed to parse llm response content: {} err: {:?}",
                content, err
            );
            return false;
        }
    };

    evaluation.violates_rules
}

struct Handler {
    model: Model,
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        let guild_id = if let Some(guild_id) = msg.guild_id {
            guild_id.get()
        } else {
            return;
        };

        let report_channel_id = match guild_id {
            145457131640848384 => 335451227028717568, // sam's bot testing server
            238666723824238602 => 1050059967337607199, // progdisc
            _ => {
                return;
            }
        };

        let report_channel = ChannelId::from(report_channel_id);

        let violates_rules = evaluate_message(&self.model, &msg.content).await;
        if violates_rules {
            // we don't want to split into an emoji or something like that
            let content_summary = UnicodeSegmentation::graphemes(msg.content.as_str(), true)
                .take(512)
                .collect::<String>();
            if let Err(err) = report_channel
                .send_message(
                    &ctx.http,
                    CreateMessage::new().embed(
                        CreateEmbed::new()
                            .field("summary", content_summary, false)
                            .field("message link", msg.link(), false)
                            .title("found violating message"),
                    ),
                )
                .await {
                error!("failed to report: {:?}", err);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_logging()
        .build()
        .await?;

    let token = env::var("DISCORD_TOKEN")?;
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::GUILD_MESSAGE_REACTIONS
        | GatewayIntents::MESSAGE_CONTENT;

    let mut client = Client::builder(&token, intents)
        .event_handler(Handler { model })
        .await?;

    client.start().await?;
    Ok(())
}
