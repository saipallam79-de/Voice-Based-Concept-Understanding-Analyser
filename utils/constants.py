"""Demonstration concepts and default speech markers."""

CONCEPTS = {
    "Machine Learning": {
        "reference": "Machine learning is a branch of artificial intelligence where systems learn patterns from data and use those patterns to make predictions or decisions. Models are trained on examples rather than being programmed with every rule.",
        "key_concepts": [
            "A branch of artificial intelligence",
            "Systems learn patterns from data",
            "Models are trained using examples",
            "Learned patterns support predictions or decisions",
            "Rules do not need to be explicitly programmed one by one",
        ],
    },
    "Deep Learning": {
        "reference": "Deep learning is a part of machine learning that uses neural networks with many layers to learn complex representations from large amounts of data. These networks can support tasks such as image recognition, speech processing, and language understanding.",
        "key_concepts": ["A subset of machine learning", "Neural networks have multiple layers", "Layers learn complex representations", "Often uses large datasets", "Supports image, speech, and language tasks"],
    },
    "Cloud Computing": {
        "reference": "Cloud computing provides on-demand access to computing resources such as servers, storage, and databases over a network. Resources can be scaled as needed and are commonly billed according to usage.",
        "key_concepts": ["Computing resources are provided on demand", "Resources include servers, storage, or databases", "Services are accessed over a network", "Capacity can scale with demand", "Usage-based billing is common"],
    },
    "Artificial Intelligence": {
        "reference": "Artificial intelligence is the field of creating computer systems that perform tasks associated with human intelligence, such as reasoning, learning, perception, and language understanding. AI includes approaches such as rule-based systems and machine learning.",
        "key_concepts": ["A field for intelligent computer systems", "Systems perform tasks associated with human intelligence", "Examples include reasoning, learning, and perception", "Language understanding is an example task", "AI includes machine learning and rule-based approaches"],
    },
    "Data Science": {
        "reference": "Data science combines statistics, computing, and domain knowledge to extract useful insights from data. A typical process collects and prepares data, analyzes it, and communicates findings to support decisions.",
        "key_concepts": ["Combines statistics and computing", "Uses domain knowledge", "Extracts insights from data", "Includes collection, preparation, and analysis", "Communicates findings to support decisions"],
    },
    "Natural Language Processing": {
        "reference": "Natural language processing is a field of artificial intelligence that enables computers to work with human language. It uses computational methods to analyze, understand, or generate text and speech.",
        "key_concepts": ["A field within artificial intelligence", "Works with human language", "Uses computational methods", "Can analyze or understand language", "Can generate text or speech"],
    },
    "Database Management": {
        "reference": "Database management organizes data in a structured system so it can be stored, queried, updated, and protected. A database management system provides tools for reliable access, integrity, and controlled sharing of data.",
        "key_concepts": ["Organizes data in a structured system", "Stores and retrieves data through queries", "Data can be updated and protected", "A DBMS provides reliable access", "Integrity and controlled sharing matter"],
    },
    "Computer Networks": {
        "reference": "A computer network connects devices so they can exchange data and share resources. Network protocols define how communication happens, while routers and switches help direct traffic between devices and networks.",
        "key_concepts": ["Connects multiple devices", "Allows data exchange and resource sharing", "Protocols define communication rules", "Routers direct traffic between networks", "Switches connect devices within a network"],
    },
    "Python": {
        "reference": "Python is a high-level, general-purpose programming language known for readable syntax. It supports multiple programming styles and has a large ecosystem of libraries used in automation, web development, data analysis, and machine learning.",
        "key_concepts": ["A high-level general-purpose language", "Readable syntax", "Supports multiple programming styles", "Has a large library ecosystem", "Used in automation, web, data, and machine learning"],
    },
}

FILLER_WORDS = (
    "um", "uh", "erm", "er", "hmm", "like", "basically", "actually",
    "literally", "you know", "I mean", "sort of", "kind of",
)

SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "m4a", "flac")