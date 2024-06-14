import { ChatOpenAI } from "@langchain/openai";
import { JsonOutputParser } from "@langchain/core/output_parsers";


async function main() {


  const model = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    temperature: 0
  });

  const chain = model.pipe(new JsonOutputParser());
  const stream = await chain.stream(
    `Output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key "name" and "population"`
  );

  for await (const chunk of stream) {
    console.log(chunk);
  }

  console.log("--- finalized inputs ---");
  // A function that operates on finalized inputs
  // rather than on an input_stream

  // A function that does not operates on input streams and breaks streaming.
  const extractCountryNames = (inputs: Record<string, any>) => {
    if (!Array.isArray(inputs.countries)) {
      return "";
    }
    return JSON.stringify(inputs.countries.map((country) => country.name));
  };

  const chain2 = model.pipe(new JsonOutputParser()).pipe(extractCountryNames);

  const stream2 = await chain2.stream(
    `output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key "name" and "population"`
  );

  for await (const chunk of stream2) {
    console.log(chunk);
  }
}

main().catch(console.error);

// https://js.langchain.com/v0.1/docs/expression_language/streaming/#working-with-input-streams