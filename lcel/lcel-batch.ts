import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";

async function main() {
  const model = new ChatOpenAI({});
  const promptTemplate = PromptTemplate.fromTemplate(
    "Tell me a joke about {topic}"
  );

  const chain = promptTemplate.pipe(model);

  const result = await chain.batch([{ topic: "bears" }, { topic: "cats" }]);

  console.log(result);
  /*
    [
      AIMessage {
        content: "Why don't bears wear shoes?\n\nBecause they have bear feet!",
      },
      AIMessage {
        content: "Why don't cats play poker in the wild?\n\nToo many cheetahs!"
      }
    ]
  */

  /*
You can also pass additional arguments to the call. 
The standard LCEL config object contains an option to set maximum concurrency, 
and an additional batch() specific config object that includes an option 
for whether or not to return exceptions instead of throwing them
*/
  console.log("--- bad model ---");
  const badModel = new ChatOpenAI({
    model: "badmodel",
  });
  const promptTemplate2 = PromptTemplate.fromTemplate(
    "Tell me a joke about {topic}"
  );

  const chain2 = promptTemplate2.pipe(badModel);

  const result2 = await chain2.batch(
    [{ topic: "bears" }, { topic: "cats" }],
    { maxConcurrency: 1 },
    { returnExceptions: true }
  );

  console.log(result2);

  /*
  Why don't bears wear shoes?
  
  Because they have bear feet!
  */
}

main().catch(console.error);

// https://js.langchain.com/v0.1/docs/expression_language/interface/#batch