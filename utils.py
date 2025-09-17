

#####Define prompts and system prompts#####
# prompt for stage 1
prompt = '''
Analyze these tweets:
{chunk_text}

Context: All tweets are from family caregivers of people with Alzheimer's/dementia. Your task is to identify specific topics WITHIN this caregiving context.

Your task:
1. Extract {topic_num} distinct topics that represent specific aspects/themes WITHIN the caregiving context
2. For each topic:
   Step 1: Create a clear label (2-4 words)
   Step 2: From the tweets, identify groups of words with high semantic similarity
   Step 3: Select EXACTLY 20 UNIQUE words where:
          - All words in the group are highly semantically similar to each other
          - Each word shares closely related meanings with others in the group
          - The semantic distance between any pair of words is minimal

IMPORTANT:
- Do NOT create topics about the general domain of Alzheimer's/dementia caregiving
- Focus on specific experiences, challenges, emotions, or activities WITHIN this context
- Exclude domain-specific terms like: Alzheimer's, dementia, caregiver, care, caring, AD, ADRD

STRICT REQUIREMENTS FOR WORD SELECTION:
- MUST have exactly 20 words per topic
- Each word MUST appear only ONCE across all topics
- NO REPETITION of words allowed
- NO placeholder words
- NO generic terms to fill space
- All words within each topic should be semantically similar to each other
- If struggling to find 20 unique words, use related terms from the tweets

Format your response EXACTLY as shown below, with NO additional text:
Topic 1:
Label: [2-4 word label]
Words: [word1], [word2], [word3], [word4], [word5], [word6], [word7], [word8], [word9], [word10], [word11], [word12], [word13], [word14], [word15], [word16], [word17], [word18], [word19], [word20]

[Continue format for Topics 2-{topic_num}]
'''

# prompt for stage 2
prompt_synthesize = '''
Analyze these words and topics extracted from tweets:
{gpt_response}

Your task:
1. Analyze these topics and synthesize them into exactly {topic_num} comprehensive, overarching topics.
2. For each topic:
   Step 1: Create a clear label (2-4 words)
   Step 2: From the provided words, identify groups of words with extremely high semantic similarity
   Step 3: Select EXACTLY 20 UNIQUE words where:
          - All words in the group are highly semantically similar to each other
          - Each word could be substituted for another in similar contexts
          - All words share closely related meanings
          - The semantic distance between any pair of words is minimal
   
Requirements:
- Use each word only ONCE across all topics
- MAXIMIZE semantic similarity between ALL words within each topic
- Focus on creating the tightest possible semantic groupings
- Avoid words that are only loosely related to the group
- Separate words with commas only

Format your response EXACTLY as shown below, with NO additional text:
Topic 1:
Label: [2-4 word label]
Words: [word1], [word2], [word3], [word4], [word5], [word6], [word7], [word8], [word9], [word10], [word11], [word12], [word13], [word14], [word15], [word16], [word17], [word18], [word19], [word20]

[Continue format for Topics 2-{topic_num}]
'''

# system prompt for stage 1
system_prompt = "You are a precise and detail-oriented assistant for topic labeling. \
Your primary directive is to extract topics from tweets while following these STRICT RULES: \
1. Each topic must have EXACTLY 20 DIFFERENT words \
2. NO WORD can be repeated within a topic's word list \
3. Every word must come from the original tweets \
4. Words must be meaningful - avoid stop words like 'the', 'and', 'or' \
5. Each word can only appear ONCE in the entire word list for each topic "

# system prompt for stage 2
system_prompt_synthesize = '''
You are a precise and detail-oriented assistant for topic labeling. 
You will be provided with 225 batches of topics extracted from tweets by caregivers of family members with dementia. 
Your primary directive is to extract topics from tweets while following these STRICT RULES: 
1. Each topic must have EXACTLY 20 DIFFERENT words 
2. NO WORD can be repeated within a topic's word list 
3. Every word must come from the given words 
4. Words must be meaningful - avoid stop words like 'the', 'and', 'or' 
5. Each word can only appear ONCE in the entire word list for each topic 
'''


#####Define functions#####

def get_topic(text, prompt, model="gpt-4o", topic_num=5):
  response = client.chat.completions.create(
  model=model,  # or another model like GPT-4 if available
  messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt.format(chunk_text=text, topic_num=topic_num)}
      ],
      temperature=0.1
    )
  return response.choices[0].message.content

def synthesize_topic(text, prompt, model="gpt-4o", topic_num=5):
  response = client.chat.completions.create(
  model=model,  # or another model like GPT-4 if available
  messages=[
          {"role": "system", "content": system_prompt_synthesize},
          {"role": "user", "content": prompt.format(gpt_response=text, topic_num=topic_num)}
      ],
      temperature=0.1
    )
  return response.choices[0].message.content

def extract_topics(data):
    # Split the string into topics based on the heading "Topic"
    topics = data.split('\n\n')
    
    # Dictionary to hold the extracted data
    extracted_data = {}
    
    for topic in topics:
        if "Topic" in topic:
            # Get the topic number
            topic_number = "Topic " + topic.split(':')[0].split(' ')[1]
            
            # Extract the label
            label_start = topic.find('Label: ') + len('Label: ')
            label_end = topic.find('\n', label_start)
            label = topic[label_start:label_end].strip()
            
            # Extract words
            words_start = topic.find('Words: ') + len('Words: ')
            words = topic[words_start:].strip().split(', ')
            
            # Populate the dictionary
            extracted_data[topic_number] = {'Label': label, 'Words': words}
    
    return extracted_data

def average_pairwise_similarity(similarity_matrix):
    A = np.array(similarity_matrix)
    n = A.shape[0]
    # Ensure matrix is square
    assert A.shape[0] == A.shape[1], "Similarity matrix must be square"
    upper_triangle_indices = np.triu_indices(n, k=1)
    upper_triangle_values = A[upper_triangle_indices]
    # Compute the average of these values
    average_similarity = np.mean(upper_triangle_values)
    
    return average_similarity


def get_mean_st(gpt_responses):
    mean_score = []
    for i in range(len(gpt_responses)):
        extracted_data = extract_topics(gpt_responses[i])
        for keys, values in extracted_data.items():
            words_list = values['Words']
            embeddings = model.encode(words_list)
            similarities = model.similarity(embeddings, embeddings)
            sim = average_pairwise_similarity(similarities)
            mean_score.append(sim)
    return np.mean(mean_score)
