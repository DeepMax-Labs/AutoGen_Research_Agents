from lib_imports import *

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


@track_agent(name="explainer")
class ExplainerAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Explainer. You collect research topic and break it down to subtopics, identifying key research 
                    questions or hypothesis. You also analyze the scope of the research topic and generate suitable 
                    research questions for an academic research. If the research topic is not good enough to generate 
                    enough scope for academic research, rephrase the topics and generate the research questions and hypothesis. 
                    You output the research topic, key research questions and hypothesis generated from the research topic.
                    """,

                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content


@track_agent(name="reviewer")
class LiteratureReviewer:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""Reviewer. You receive the research topic and subtopics from the Explainer Agent and then gather existing 
                    research, journal articles, papers, and other relevant literature from trusted sources. Summarize 
                    these findings, identify research gaps and organize the information. Produce this summary in the 
                    format of a standard academic literature review, highlighting important existing research and 
                    citing papers used in the review.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content
     
@track_agent(name="formulator")
class FormulationAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""Formulator. You refine the research questions and subtopics from the Explainer Agent based on the 
                    Literature Review done by the Reviewer and define the exact question the paper will address. If a 
                    hypothesis is required, propose one. Your output is the finalized research question and hypothesis.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content
  
@track_agent(name="director")
class MethodAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""Director. You receive the final research question from Formulator and literature review from 
                    Reviewer and suggest appropriate research methods (qualitative, quantitative, or mixed) based on 
                    the research question. Also, outline data collection methods, sample sizes, tools, and techniques 
                    necessary for this research. Your output is the proposed research methodology, proposing necessary 
                    data collection methods, analysis plans, and tools for the research.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content
  
@track_agent(name="collector")
class CollectorAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""Collector. You receive the proposed methodology from Director Agent and gather real data from the 
                    web or simulate the data related to the research question. Ensure that the data are from verifiable 
                    and reliable sources, and are linked to the research being conducted. Your output are simulated or 
                    recommended datasets for analysis.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content
  
@track_agent(name="analyst")
class DataAnalyst:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""Analyst. You receive data and analyze the collected data using appropriate statistical or 
                    qualitative analysis methods. For quantitative research, you could run statistical tests; for 
                    qualitative research, you would conduct thematic analysis. Your output are charts, graphs, 
                    insights, detected trends and predictions (if there are any). The analysis should be dynamic, 
                    customized to suit the requirement of the research field lies in.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content

@track_agent(name="interpreter")
class DiscussionAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""Interpreter. You collect data analysis results from Analyst Agent and literature review from 
                    Reviewer. You should interpret the analysis results, compares them with existing research in the 
                    review, and provides insights into how the findings answer the research question or support/refute 
                    the hypothesis, provided by the Formulator. Your output is a discussion of findings, implications, 
                    and potential limitations.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content
 
@track_agent(name="recommender")
class RecommendationAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""Recommender. You collect discussion of findings, implications and potential limitations from the 
                    Interpreter Agent and summarize the research findings, proposes future research directions, and 
                    provides recommendations or solutions based on the findings. Your output is conclusion and 
                    recommendation section of the paper.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content

@track_agent(name="formatter")
class CitationAgent:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""Formatter. You collect different parts of the research paper and ensure the entire research paper 
                    is formatted according to the chosen academic style (APA, MLA, etc.). You also ensure proper 
                    citation process, collecting and formatting citations from literature sources. Your output is a 
                    fully formatted academic paper with citations and references.
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content

@track_agent(name="qaengineer")
class QAEngineer:
    def completion(self, prompt: str):
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content":"""QAEngineer. You collect a draft of an academic research paper and perform proofreading, plagiarism 
                    checks, grammar and style analysis, and overall coherence evaluation. Your output is a PDF document 
                    containing a refined, high-quality research paper. 
                    """,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16000,
        )
        return res.choices[0].message.content


                           
