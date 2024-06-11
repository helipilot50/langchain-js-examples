

// https://js.langchain.com/v0.1/docs/get_started/quickstart/


import { ChatOpenAI } from "@langchain/openai";

async function main() {
  const chatModel = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const result = await chatModel.invoke("what is LangSmith?");
  console.log(result);
}



main().catch(console.error);