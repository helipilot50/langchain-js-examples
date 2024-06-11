

// https://js.langchain.com/v0.1/docs/get_started/quickstart/


import { ChatOpenAI, } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";


async function main() {
  const chatModel = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  let result: any = await chatModel.invoke("what is LangSmith?");
  console.log(result);


  // We can also guide it's response with a prompt template. Prompt templates are used to convert raw user input to a better input to the LLM.

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a world class technical documentation writer."],
    ["user", "{input}"],
  ]);

  // We can now combine these into a simple LLM chain:
  const chain = prompt.pipe(chatModel);

  // We can now invoke it and ask the question:
  result = await chain.invoke({
    input: "what is LangSmith?",
  });

  console.log(result);

  // The output of a ChatModel (and therefore, of this chain) is a message. However, it's often much more convenient to work with strings. Let's add a simple output parser to convert the chat message to a string.

  const outputParser = new StringOutputParser();

  const llmChain = prompt.pipe(chatModel).pipe(outputParser);

  result = await llmChain.invoke({
    input: "what is LangSmith?",
  });

  console.log(result);

}

main().catch(console.error);