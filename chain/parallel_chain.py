from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()


model1 =  ChatGoogleGenerativeAI(model ='gemini-2.5-flash')

model2 = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template= 'Generate short and  simple note  from the following text \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Genrate 5 short question answer from follwing text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='marge the provide notes and quiz a single document\n notes -> {notes} and quiz ->{quiz}',
    input_variables=['notes','quiz']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
     'notes': prompt1 | model1 | parser,
     'quiz' : prompt2 | model2 | parser
})

marge_chain = prompt3 | model1 | parser


chain = parallel_chain | marge_chain

text  ="""According to The Unicode Standard
Plain text is a pure sequence of character codes; plain Un-encoded text is therefore a sequence of Unicode character codes.
In contrast, styled text, also known as rich text, is any text representation containing plain text plus added information such as a language identifier, font size, color, hypertext links, and so on.
SGML, RTF, HTML, XML, and TeX are examples of rich text fully represented as plain text streams, interspersing plain text data with sequences of characters that represent the additional data structures.
According to other definitions, however, files that contain markup or other meta-data are generally considered plain text, so long as the markup is also in a directly human-readable form (as in HTML, XML, and so on). Thus, representations such as SGML, RTF, HTML, XML, wiki markup, and TeX, as well as nearly all programming language source code files, are considered plain text. The particular content is irrelevant to whether a file is plain text. For example, an SVG file can express drawings or even bitmapped graphics, but is still plain text
The use of plain text rather than binary files enables files to survive much better "in the wild", in part by making them largely immune to computer architecture incompatibilities. For example, with all data encoded as UTF-8 text, all the problems of endianness can be avoided."""

result = chain.invoke({'text':text})
print(result)

