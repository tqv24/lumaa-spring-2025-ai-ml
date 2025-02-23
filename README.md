# Content-Based Book Recommendation System

## Overview
This project is a **content-based recommendation system** that suggests relevant books or movies based on a user's input description. It leverages **text similarity** techniques, utilizing **TF-IDF** and **FastEmbed** to compute the similarity between the user's query and dataset items.

**Key Features:**
- Supports **TF-IDF** and **FastEmbed** models for text embedding.
- Uses **cosine similarity** to rank recommendations.
- Works with **a small dataset** (100–500 items) for quick execution.
- **Command-line interface** (CLI) for easy interaction.
- **Modular design** for extensibility (e.g., different datasets, hybrid models).

---

## Dataset
The dataset (`data.csv`) contains a list of **movies/books with their title, author, genres, and description**.
- **`title`** → Name of the book/movie.
- **`author`** → Author or director.
- **`generes`** → Genres associated with the item.
- **`description`** → Brief description or plot summary.

## Setup and Run code

### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3.Run the Recommendation System**
```bash
python app.py 
```

## Expected Results:
```yaml
Enter your query or type 'exit' to close the app: I love detective book that are thriller and mysterious.
Choose model (tfidf/fastembed): tfidf

- Find Me: A Novel by André Aciman | Genre: Fiction , LGBT , Gay | Similarity Score: 0.1255
- Once Missed (A Riley Paige Mystery—Book 16) by Blake Pierce | Genre: Fiction , Mystery &amp, Detective , Women Sleuths | Similarity Score: 0.1175
- Watching (The Making of Riley Paige—Book 1) by Blake Pierce | Genre: Fiction , Mystery &amp, Detective , Women Sleuths | Similarity Score: 0.1144
- And Then There Were None by Agatha Christie | Genre: Fiction , Mystery &amp, Detective , Cozy , General | Similarity Score: 0.1095
- Blood Runs Cold: A completely unputdownable mystery and suspense thriller by Dylan Young | Genre: Fiction , Mystery &amp, Detective , Police Procedural | Similarity Score: 0.1047
```

## Project structure:
```bash
/your_repo/
│── app.py                 # Main script
│── data.csv               # Dataset
│── requirements.txt        # Dependencies
│── README.md               # Documentation
│── demo.md                 # Demo video link
```

## Salary Expectation
- Expected Monthly Salary: $XXXX (Replace with your amount)