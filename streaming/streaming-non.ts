import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";



import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import type { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";


async function main() {

  const model = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    temperature: 0
  });

  const template = `Answer the question based only on the following context:
{context}

Question: {question}
`;
  const prompt = ChatPromptTemplate.fromTemplate(template);

  const vectorstore = await MemoryVectorStore.fromTexts(
    ["mitochondria is the powerhouse of the cell", "buildings are made of brick"],
    [{}, {}],
    new OpenAIEmbeddings()
  );

  const retriever = vectorstore.asRetriever();

  const formatDocs = (docs: Document[]) => {
    return docs.map((doc) => doc.pageContent).join("\n-----\n");
  };

  const retrievalChain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocs),
      question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
  ]);

  const stream = await retrievalChain.stream(
    "What is the powerhouse of the cell?"
  );

  for await (const chunk of stream) {
    console.log(`${chunk}|`);
  }
}

main().catch(console.error);

// https://js.langchain.com/v0.1/docs/expression_language/streaming/#non-streaming-components