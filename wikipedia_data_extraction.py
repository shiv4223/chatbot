import wikipediaapi
import chromadb

wiki = wikipediaapi.Wikipedia("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",'en')

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="wiki_knowledge")

# List of Wikipedia topics to fetch
main_topics = ["Artificial intelligence", "Machine learning", "Coding", "Physics", "Maths", "chemistry", "SQL"]

def extract_wikipedia_summary(topic):
    """Fetch summary of a Wikipedia page."""
    page = wiki.page(topic)
    if page.exists():
        return page.summary
    return None

def get_related_topics(topic, depth=1):
    """
    Fetch subtopics (linked Wikipedia pages) up to a given depth.
    """
    page = wiki.page(topic)
    if not page.exists():
        return []

    related_topics = list(page.links.keys())
    return related_topics[:50]  

def extract_wikipedia_details(topic):
    """
    Extracts summary and key section details from a Wikipedia page.
    """
    page = wiki.page(topic)
    if not page.exists():
        return None

    # Extract full summary
    data = {"title": topic, "summary": page.summary, "sections": {}}

    # Extract section-wise content
    for section in page.sections:
        data["sections"][section.title] = section.text

    return data

# Extract subtopics dynamically
detailed_topics = {'Artificial intelligence': ['15.ai',
  '2001: A Space Odyssey',
  '2001: A Space Odyssey (novel)',
  '2024 Indian general election',
  '3D optical data storage',
  'A* search algorithm',
  'A.I. Artificial Intelligence',
  'AAAI',
  'ABB',
  'ACM Computing Classification System',
  'ACM Conference on Fairness, Accountability, and Transparency',
  'ACM SIGPLAN Notices',
  'AI',
  'AI (disambiguation)',
  'AI Safety Institute (United Kingdom)'],
 'Coding': ['Channel coding',
  'Code',
  'Coding (social sciences)',
  'Coding (therapy)',
  'Coding strand',
  'Coding theory',
  'Computer programming',
  'Entropy encoding',
  'Legal coding',
  'Line coding',
  'Medical coding',
  'Number coding in Metro Manila',
  'Queer coding',
  'Source coding',
  'Transform coding'],
 'Physics': ['A Greek–English Lexicon',
  'A priori and a posteriori',
  'About.com',
  'Academic discipline',
  'Accelerator physics',
  'Acoustic engineering',
  'Acoustical Society of America',
  'Acoustics',
  'Action (physics)',
  'Active learning',
  'AdS/CFT correspondence',
  'Agrophysics',
  'Al-Kindi',
  'Albert Einstein',
  'Amber'],
 'Maths': ["A Mathematician's Apology",
  'Abel Prize',
  'Abstract algebra',
  'Abstraction (mathematics)',
  'Accountant',
  'Actually infinite',
  'Actuary',
  'Addison-Wesley Publishing Company',
  'Addition',
  'Adrien-Marie Legendre',
  'Advances in Mathematics',
  'Aesthetic',
  'Affine geometry',
  'Al-Jabr',
  'Al-Khwarizmi'],
 'chemistry': ['4-Hydroxybutanal',
  'Abū al-Rayhān al-Bīrūnī',
  'Acid dissociation constant',
  'Acid–base reaction',
  'Actinide chemistry',
  'Activation energy',
  'Agricultural chemistry',
  'Agrochemistry',
  'Air (classical element)',
  'Alchemists',
  'Alchemy',
  'Alchemy and chemistry in Islam',
  'Alcohol (chemistry)',
  'Aldehyde']}

# Fetch and store Wikipedia details in ChromaDB
topic_names = set()
for topic, subtopics in detailed_topics.items():
    print(f"Fetching details for {topic} and its subtopics...")

    # Fetch main topic details
    main_content = extract_wikipedia_details(topic)
    if main_content:
        collection.add(
            ids=[topic.replace(" ", "_")],
            documents=[main_content["summary"]]
        )

    # Fetch subtopic details
    for subtopic in subtopics:
        if subtopic in topic_names:
            continue
        topic_names.add(subtopic)
        sub_content = extract_wikipedia_details(subtopic)
        if sub_content:
            
            collection.add(
                ids=[subtopic.replace(" ", "_")],
                documents=[sub_content["summary"]]
            )

print("Wikipedia knowledge stored in ChromaDB ✅")
print(len(topic_names))





