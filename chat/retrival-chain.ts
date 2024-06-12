import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { createRetrievalChain } from "langchain/chains/retrieval";


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

  // If we wanted to, we could run this ourselves by passing in documents directly:
  let result: any = await documentChain.invoke({
    input: "what is LangSmith?",
    context: [
      new Document({
        pageContent:
          "LangSmith is a platform for building production-grade LLM applications.",
      }),
    ],
  });

  console.log('direct from documents', result);

  // However, we want the documents to first come from the retriever we just set up. 
  // That way, for a given question we can use the retriever to 
  // dynamically select the most relevant documents and pass those in.


  const retriever = vectorstore.asRetriever();

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  // We can now invoke this chain. This returns an object - the response from the LLM is in the answer key:
  result = await retrievalChain.invoke({
    input: "what is LangSmith?",
  });

  console.log('using retrival chain', result.answer);

}

main().catch(console.error);