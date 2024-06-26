import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";

async function main() {
  const model = new ChatOpenAI({});
  const promptTemplate = PromptTemplate.fromTemplate(
    "Tell me a joke about {topic}"
  );

  // You can also create a chain using an array of runnables
  const chain = RunnableSequence.from([promptTemplate, model]);

  const result = await chain.invoke({ topic: "bears" });

  console.log(result);
  /*
    AIMessage {
      content: "Why don't bears wear shoes?\n\nBecause they have bear feet!",
    }
  */
}

main().catch(console.error);

// https://js.langchain.com/v0.1/docs/expression_language/interface/#invoke