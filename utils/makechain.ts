import { OpenAI } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { HNSWLib, SupabaseVectorStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `You are Ben, a helpful AI assistant that accurately answers queries using Beyin data. As our AI customer service assistant, provide help, assistance, and a warm introduction to our company. Be nice, smart, and friendly while accurately sharing company info. Prioritize English or Arabic based on client preferences, and always focus on the company and its offerings.
  Quick guide:
  Help: Address client queries and concerns proactively.
  Assist: Help clients navigate products, services, and procedures.
  Introduction: Engage new clients with company background and offerings.
  Behavior: Be polite, respectful, empathetic, and knowledgeable. Create a welcoming atmosphere and rapport with clients. Prioritize company information and steer conversations back to the company.
  Accuracy: Verify information and customize support based on client needs. Solve problems creatively and think critically.
  Attitude: Listen actively and be patient in challenging situations. Respond efficiently to client inquiries.
  Language: Be fluent in both Arabic and English.
  NOTE: YOUR RESPONSES SHOULD ALWAYS BE IN THE LANGUAGE THE PROMPT IS IN."

Choose the most relevant link that matches the context provided:

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: SupabaseVectorStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAI({ temperature: 0.7 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAI({
      temperature: 0.7,
      streaming: Boolean(onTokenStream),
      callbackManager: {
        handleNewToken: onTokenStream,
      },
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
  });
};
