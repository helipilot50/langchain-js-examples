import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createRetrieverTool } from "langchain/tools/retriever";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { pull } from "langchain/hub";
import { ChatOpenAI, } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  createOpenAIFunctionsAgent,
  AgentExecutor
} from "langchain/agents";
import { HumanMessage, AIMessage } from "@langchain/core/messages";


async function main() {
  const loader = new CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/user_guide"
  );
  const docs = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings();
  const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  const retriever = vectorstore.asRetriever();

  /*
  One of the first things to do when building an agent is to decide what tools it should have access to. For this example, we will give the agent access two tools:
  
  - The retriever we just created. This will let it easily answer questions about LangSmith
  - A search tool. This will let it easily answer questions that require up to date information.
  */


  // Create a retriever tool
  const retrieverTool = await createRetrieverTool(retriever, {
    name: "langsmith_search",
    description:
      "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
  });

  /*
  The search tool that we will use is Tavily. 
  This will require you to create an API key (they have generous free tier).
  */

  // Create a search tool
  const searchTool = new TavilySearchResults();

  // We can now create a list of the tools we want to work with:
  const tools = [retrieverTool, searchTool];

  // create an agent to use them and an executor to run the agent.


  // Get the prompt to use - you can modify this!
  // If you want to see the prompt in full, you can at:
  // https://smith.langchain.com/hub/hwchase17/openai-functions-agent
  const agentPrompt = await pull<ChatPromptTemplate>(
    "hwchase17/openai-functions-agent"
  );

  const agentModel = new ChatOpenAI({
    model: "gpt-3.5-turbo-1106",
    temperature: 0,
  });

  const agent = await createOpenAIFunctionsAgent({
    llm: agentModel,
    tools,
    prompt: agentPrompt,
  });

  const agentExecutor = new AgentExecutor({
    agent,
    tools,
    // verbose: true,
  });

  // invoke the agent and see how it responds! We can ask it questions about LangSmith:
  let agentResult: any = await agentExecutor.invoke({
    input: "how can LangSmith help with testing?",
  });

  console.log('--- Invoke agent about LangSmith ---\n', agentResult.output);

  // We can ask it about the weather:
  agentResult = await agentExecutor.invoke({
    input: "what is the weather in SF?",
  });

  console.log('--- Invoke agent about the weather ---\n', agentResult.output);

  // conversation with agent

  agentResult = await agentExecutor.invoke({
    chat_history: [
      new HumanMessage("Can LangSmith help test my LLM applications?"),
      new AIMessage("Yes!"),
    ],
    input: "Tell me how",
  });

  console.log('--- conversation ---\n', agentResult.output);

}


main().catch(console.error);