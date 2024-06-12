import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, } from "@langchain/openai";
import { Document } from "@langchain/core/documents";


async function main() {

  // First, we need to load the data that we want to index. We'll use a document loader that uses 
  // the popular Cheerio npm package as a peer dependency to parse data from webpages. 
  const loader = new CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/user_guide"
  );

  const docs = await loader.load();
  console.log('Number of docs', docs.length);
  console.log('First doc content size', docs[0].pageContent.length);

  // We can split the document into more manageable chunks to get around this limitation 
  // and to reduce the amount of distraction to the model using a text splitter:

  const splitter = new RecursiveCharacterTextSplitter();

  const splitDocs = await splitter.splitDocuments(docs);

  console.log('Number of split docs', splitDocs.length);
  console.log('First split doc content size', splitDocs[0].pageContent.length);


  // Now, we can use this embedding model to ingest documents into a vectorstore.
  // The LangChain vectorstore class will automatically prepare each raw document using the embeddings model.

  const embeddings = new OpenAIEmbeddings();
  const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  // Now that we have this data indexed in a vectorstore, we will create a retrieval chain. 
  // This chain will take an incoming question, look up relevant documents, 
  // then pass those documents along with the original question into an LLM and ask it to answer the original question.

  const chatModel = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const prompt =
    ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
  });

  const result = await documentChain.invoke({
    input: "what is LangSmith?",
    context: [
      new Document({
        pageContent:
          "LangSmith is a platform for building production-grade LLM applications.",
      }),
    ],
  });

  console.log(result);
}

main().catch(console.error);