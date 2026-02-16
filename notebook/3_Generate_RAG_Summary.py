import os
import json
import pandas as pd
import glob
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
import concurrent.futures
from ast import literal_eval

# Load environment variables
load_dotenv()

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

# Define connection to your Neo4j instance
url = os.environ["NEO4J_URI"]
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize Neo4jVector store for vector embeddings
neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    node_label="Chunk",
    embedding_node_property="embedding",
    text_node_properties=["text"],
    url=url,
    username=username,
    password=password,
)

# Initialize the OpenAI model
model = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)

# Ensure log directory exists
log_dir = "./logs/v2"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def retrieve_candidate(q, country, sector, neo4j_vector, model):
    # Perform the similarity search with a filter
    a = neo4j_vector.similarity_search_with_score(
        q, k=1, filter={"geography": country, "sector": sector}
    )

    for i in a:
        doc, score = i
        doc_id = doc.metadata["document_id"]

        return doc.page_content


# Function to process a single step_1_query
def process_query(q, country, sector, neo4j_vector, model):
    all_chunks = []
    document_ids = set()

    # Perform the similarity search with a filter
    a = neo4j_vector.similarity_search_with_score(
        q, k=1000, filter={"geography": country, "sector": sector}
    )

    for i in a:
        doc, score = i
        doc_id = doc.metadata["document_id"]

        if doc_id not in document_ids and len(document_ids) < 3:
            all_chunks.append(doc.page_content)
            document_ids.add(doc_id)

        if len(document_ids) == 3:
            break

    # Prepare the instruction and input text for the LLM
    instruction = (
        "Generate a single coherent paragraph based on the following time series analysis and related content. "
        "Ensure that the writing style of the generated paragraph is a blend of the three other chunks, "
        "while incorporating the insights from the time series analysis. Do not mention the name of any policy, plan, act, or refer to 'time series analysis' explicitly."
    )

    input_text = (
        instruction
        + "\n\nTime Series Analysis Chunk:\n"
        + q
        + "\n\nOther Related Chunks:\n"
        + "\n".join(all_chunks)
    )

    # Invoke the LLM to generate new chunks based on the input and the instruction
    new_chunk = model.invoke(input_text)

    return new_chunk.content


# Generate new chunks in parallel
def generate_chunks_in_parallel(step_1_queries, country, sector, neo4j_vector, model):
    all_new_chunks = []  # Store all new chunks

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        future_to_query = {
            executor.submit(process_query, q, country, sector, neo4j_vector, model): q
            for q in step_1_queries
        }
        # future_to_query = {
        #     executor.submit(
        #         retrieve_candidate, q, country, sector, neo4j_vector, model
        #     ): q
        #     for q in step_1_queries
        # }

        # Collect the results as they are completed
        for future in concurrent.futures.as_completed(future_to_query):
            try:
                result = future.result()
                all_new_chunks.append(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    return all_new_chunks


# Load the CSV file and limit to the first 20 rows
df = pd.read_csv("csv/test_with_summary.csv")

# Load table names and metadata
table_names = dict()
table_titles = list()

for file in glob.glob("../owid/*/*.meta.json"):
    with open(file) as f:
        obj = json.load(f)
    if obj.get("title", None):
        title = obj.get("title").lower()
    else:
        title = obj.get("dataset").get("title").lower()

    table_names[title] = file
    table_titles.append(title)


# Function to retrieve data as a string based on filters
def retrieve_data_as_string(df, location, input_year):
    # Check if 'country' or 'location' column exists and apply filtering
    if "country" in df.columns:
        if location == "European Union":
            # Special condition for European Union
            filtered_df = df[df["country"].str.contains("European Union", na=False)]
        else:
            # General case for country filtering
            filtered_df = df[df["country"] == location]
    elif "location" in df.columns:
        # General case for location filtering
        filtered_df = df[df["location"] == location]
    else:
        # If neither 'country' nor 'location' columns exist, return empty string
        return ""

    # Further filter by year
    filtered_df = filtered_df[
        (filtered_df["year"] >= input_year - 10) & (filtered_df["year"] <= input_year)
    ]

    # Convert the filtered DataFrame to a string representation
    result_string = filtered_df.to_string(index=False)
    return result_string


# List to hold new summaries
new_summaries = []

i = 0

# Iterate over each row in the dataframe
for index, row in tqdm(df.iterrows(), total=len(df)):
    i += 1
    if i < 93:
        continue
    a = f'Targeted Sector: {row["Sector"]}\nPolicy Instrument: {row["Instrument"]}\nKeywords: {row["Keyword"]}\nTopics: {row["Topic/Response"]}\nHazards: {row["Hazard"]}'
    country = row["Geography"]
    region = row["region"]
    year = int(row["Last event in timeline"].split("-")[0])
    sector = row["Sector"]
    doc_id = row["doc_id"]  # Assuming the row contains a column "doc_id"

    print(a)

    # Invoke the model to get the table name and sector
    ans = model.invoke(
        f"""
        Based on this table's names: {table_titles}, I want you to select all the table's names from the list that might be directly related to the creation of the policy with the following metadata.

        POLICY METADATA:
        {a}

        Return in list with no explaination.
        Begin:
        """
    )

    # Convert the response to a list
    result = set(literal_eval(ans.content))
    step_1_queries = []

    # Loop over each result (table names and sectors)
    for r in result:
        text = "DATASET:\n"
        file_path = table_names[r.lower()]

        with open(file_path) as file:
            obj = json.load(file)

        desc = obj["dataset"].get("description", None)
        text = text + f"Title: {r}\n"
        if desc is not None:
            text = text + f"Description: {desc}\n"

        text = text + f"Focused Country Data: {country}\n"
        inp = pd.read_csv(file_path.replace(".meta.json", ".csv"))
        text = text + retrieve_data_as_string(inp, country, year)

        text = text + f"{region} Data:\n"
        text = text + retrieve_data_as_string(inp, region, year)

        text = text + "World Data:\n"
        text = text + retrieve_data_as_string(inp, "World", year)

        try:
            ans = model.invoke(
                f"""
            You are policy analyzer expert.
            You are to analyse the following data, but keep in mind that you are focused at {sector} aspects.
            Here is the data you will be analysed.

            {text}

            ----

            Output only one paragraph with no explanation.
            Begin:
            """
            )

            step_1_queries.append(ans.content)
        except:
            continue

    all_new_chunks = generate_chunks_in_parallel(
        step_1_queries, country, sector, neo4j_vector, model
    )

    all_new_chunks = list(set(all_new_chunks))

    # Final summary generation
    final_ans = model.invoke(
        f"""
    You are policy analyzer expert.
    You are to summarize these chunks of information to one paragraph:
    
    {'[NEW_CHUNK]'.join(all_new_chunks)}
    
    ----
    
    Output only one paragraph with no explanation.
    Begin:
    """
    )

    # Save the generated output to a JSON file in the ../logs/v2/{doc_id}.json
    log_data = {
        "doc_id": doc_id,
        "table_selection_output": list(result),
        "new_chunks": all_new_chunks,
        "final_summary": final_ans.content,
    }
    with open(f"{log_dir}/{doc_id}.json", "w") as log_file:
        json.dump(log_data, log_file, indent=4)

    # Append the generated summary to the list
    # new_summaries.append(final_ans.content)

# # Add the new summaries to the dataframe as a new column
# df["RAG_v1_summary"] = new_summaries

# # Save the dataframe to a new CSV file
# df.to_csv("csv/test_with_summary_1.csv", index=False)

# print("New CSV with summaries saved as 'csv/test_with_summary_1.'")
