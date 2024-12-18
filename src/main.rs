use anyhow::Result;
use async_trait::async_trait;
use mistralrs::Constraint::JsonSchema;
use mistralrs::{Model, RequestBuilder, TextMessageRole, TextModelBuilder, TokenSource};
use serde::Deserialize;
use serde_json::json;
use serenity::all::{ChannelId, Context, CreateEmbed, CreateEmbedFooter, CreateMessage, EventHandler, GatewayIntents, Message};
use serenity::Client;
use std::env;
use std::time::Instant;
use tracing::{error, info, warn};
use unicode_segmentation::UnicodeSegmentation;

const SELF_ID: u64 = 1314997214866571284;

async fn generate_model() -> Result<Model> {
    TextModelBuilder::new("meta-llama/Llama-3.2-3B-Instruct")
        .with_logging()
        .with_token_source(TokenSource::EnvVar("HUGGING_FACE_HUB_TOKEN".to_string()))
        .build()
        .await
}

#[derive(Deserialize, Debug)]
struct Evaluation {
    pub violates_rules: bool,
}

async fn evaluate_message(model: &Model, content: &str) -> bool {
    let request = RequestBuilder::new()
        .set_sampler_temperature(0.)
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
            You will be provided messages to evaluate whether user breaks these rules and you will respond true if it breaks rules, false if it doesn't.\
            Don't try to make indirect connections to rules.",
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

    // println!("RESPONSE: {:?}", response);

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
        info!("received message event");
        let guild_id = if let Some(guild_id) = msg.guild_id {
            guild_id.get()
        } else {
            warn!("received message event with no guild id {:?}", msg);
            return;
        };

        if msg.content.is_empty() || msg.author.bot || msg.author.id.get() == SELF_ID {
            info!("message is likely from a bot, skipping...");
            return;
        }

        let report_channel_id = match guild_id {
            145457131640848384 => 335451227028717568, // sam's bot testing server
            238666723824238602 => 1315930244682743839, // progdisc
            _ => {
                return;
            }
        };

        let report_channel = ChannelId::from(report_channel_id);

        let before = Instant::now();
        let violates_rules = evaluate_message(&self.model, &msg.content).await;
        let after = Instant::now();

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
                            .footer(CreateEmbedFooter::new(format!("took {:.2} seconds", after.duration_since(before).as_secs_f64())))
                            .title("found violating message"),
                    ),
                )
                .await {
                error!("failed to report: {:?}", err);
            }
        }
    }
}

#[derive(Deserialize)]
struct TestCase {
    content: String,
    expected_result: bool,
}

async fn run_tests() -> Result<()> {
    let model = generate_model().await?;
    let cases: Vec<TestCase> = serde_json::from_str(include_str!("../messages.json"))?;
    let mut times = vec![];

    let mut mispredictions = 0;
    for case in cases.iter() {
        let before = Instant::now();
        let result = evaluate_message(&model, &case.content).await;
        let after = Instant::now();

        times.push(after.duration_since(before).as_secs_f64());
        if case.expected_result != result {
            println!("---");
            println!("case failed with expected result {}", case.expected_result);
            println!("{}", case.content);
            println!("---");
            mispredictions += 1;
        }
    }

    let average_time = times.iter().sum::<f64>() / times.len() as f64;
    println!("average time: {}s", average_time);

    println!("results: {}/{}", cases.len() - mispredictions, cases.len());

    assert_eq!(mispredictions, 0);

    Ok(())
}

async fn run_bot() -> Result<()> {
    let model = generate_model().await?;
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

#[tokio::main]
async fn main() -> Result<()> {
    // run_tests().await;
    run_bot().await?;
    Ok(())
}
