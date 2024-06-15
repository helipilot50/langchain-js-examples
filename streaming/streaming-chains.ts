import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

async function main() {


  const model = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    temperature: 0
  });

  const prompt = ChatPromptTemplate.fromTemplate("Tell me a joke about {topic}");

  const parser = new StringOutputParser();

  const chain = prompt.pipe(model).pipe(parser);

  const stream = await chain.stream({
    topic: "parrot",
  });

  for await (const chunk of stream) {
    console.log(`${chunk}|`);
  }
}

main().catch(console.error);

// https://js.langchain.com/v0.1/docs/expression_language/streaming/#chains