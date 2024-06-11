

// https://js.langchain.com/v0.1/docs/get_started/quickstart/


import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});