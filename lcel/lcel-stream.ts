import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";

async function main() {
  const model = new ChatOpenAI({});
  const promptTemplate = PromptTemplate.fromTemplate(
    "Tell me a joke about {topic}"
  );

  const chain = promptTemplate.pipe(model);

  const stream = await chain.stream({ topic: "old white men" });

  // Each chunk has the same interface as a chat message
  for await (const chunk of stream) {
    console.log(chunk?.content);
  }


}

main().catch(console.error);


// https://js.langchain.com/v0.1/docs/expression_language/interface/#stream