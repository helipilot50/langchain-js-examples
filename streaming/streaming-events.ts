import { ChatOpenAI } from "@langchain/openai";
import { JsonOutputParser } from "@langchain/core/output_parsers";

async function main() {
  const model = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    temperature: 0
  });

  console.log('--- chat ---');
  // chat model
  let events = [];

  let eventStream = await model.streamEvents("hello", { version: "v1" });

  for await (const event of eventStream) {
    events.push(event);
  }

  console.log('events count: ', events.length);
  console.log('3 events: ', events.slice(0, 3));

  // chain

  console.log('--- chain ---');

  let chain = model.pipe(new JsonOutputParser());
  eventStream = await chain.streamEvents(
    `Output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key "name" and "population"`,
    { version: "v1" }
  );

  events = [];
  for await (const event of eventStream) {
    events.push(event);
  }

  console.log('events count: ', events.length);
  console.log('3 events: ', events.slice(0, 3));

  console.log('--- output the stream events ---');
  // output the stream events from the model
  let eventCount = 0;

  eventStream = await chain.streamEvents(
    `Output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key "name" and "population"`,
    { version: "v1" }
  );

  for await (const event of eventStream) {
    // Truncate the output
    if (eventCount > 30) {
      continue;
    }
    const eventType = event.event;
    if (eventType === "on_llm_stream") {
      console.log(`Chat model chunk: ${event.data.chunk.message.content}`);
    } else if (eventType === "on_parser_stream") {
      console.log(`Parser chunk: ${JSON.stringify(event.data.chunk)}`);
    }
    eventCount += 1;
  }

  console.log('--- filtering events ---');
  // Filtering events
  // You can filter by either component name, component tags or component type.

  // filtering by name
  console.log('\t by name');
  let chain2 = model
    .withConfig({ runName: "model" })
    .pipe(new JsonOutputParser().withConfig({ runName: "my_parser" }));

  eventStream = await chain2.streamEvents(
    `Output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key "name" and "population"`,
    { version: "v1" },
    { includeNames: ["my_parser"] }
  );

  eventCount = 0;

  for await (const event of eventStream) {
    // Truncate the output
    if (eventCount > 10) {
      continue;
    }
    console.log(event);
    eventCount += 1;
  }

  // by type
  console.log('\t by type');
  chain2 = model
    .withConfig({ runName: "model" })
    .pipe(new JsonOutputParser().withConfig({ runName: "my_parser" }));

  eventStream = await chain.streamEvents(
    `Output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key "name" and "population"`,
    { version: "v1" },
    { includeTypes: ["llm"] }  // by type
  );

  eventCount = 0;

  for await (const event of eventStream) {
    // Truncate the output
    if (eventCount > 10) {
      continue;
    }
    console.log(event);
    eventCount += 1;
  }


  // by tags
  console.log('\t by tags');
  chain2 = model
    .pipe(new JsonOutputParser().withConfig({ runName: "my_parser" }))
    .withConfig({ tags: ["my_chain"] });

  eventStream = await chain.streamEvents(
    `Output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key "name" and "population"`,
    { version: "v1" },
    { includeTags: ["my_chain"] }
  );

  eventCount = 0;

  for await (const event of eventStream) {
    // Truncate the output
    if (eventCount > 10) {
      continue;
    }
    console.log(event);
    eventCount += 1;
  }

}

main().catch(console.error);




// https://js.langchain.com/v0.1/docs/expression_language/streaming/#using-stream-events