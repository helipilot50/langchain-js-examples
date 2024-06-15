import { ChatOpenAI } from "@langchain/openai";
async function main() {


  const model = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    temperature: 0
  });

  const stream = await model.stream("Hello! Tell me about yourself.");
  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
    console.log(`${chunk.content}|`);
  }
}

main().catch(console.error);

// https://js.langchain.com/v0.1/docs/expression_language/streaming/#llms-and-chat-models