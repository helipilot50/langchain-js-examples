import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, } from "@langchain/openai";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
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

  const chatModel = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });


  // This chain will take in the most recent input (input) 
  // and the conversation history (chat_history) and use an LLM to generate a search query.
  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);

  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever,
    rephrasePrompt: historyAwarePrompt,
  });


  const chatHistory = [
    new HumanMessage("Can LangSmith help test my LLM applications?"),
    new AIMessage("Yes!"),
  ];

  // We can test this "history aware retriever" out by creating a situation where the user is asking a follow up question:

  let result: any = await historyAwareRetrieverChain.invoke({
    chat_history: chatHistory,
    input: "Tell me how!",
  });

  console.log('Hostory aware chain:\n', result);

  // with this retriever, we can create a new chain to continue the conversation with these retrieved documents in mind:
  const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the below context:\n\n{context}",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  const historyAwareCombineDocsChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: historyAwareRetrievalPrompt,
  });

  const conversationalRetrievalChain = await createRetrievalChain({
    retriever: historyAwareRetrieverChain,
    combineDocsChain: historyAwareCombineDocsChain,
  });


  result = await conversationalRetrievalChain.invoke({
    chat_history: [
      new HumanMessage("Can LangSmith help test my LLM applications?"),
      new AIMessage("Yes!"),
    ],
    input: "tell me how",
  });

  console.log('history Aware CombineDocs Chain:\n', result.answer);

}

main();