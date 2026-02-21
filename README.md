# AI Engineering — Brochure Generator

Scrapes a company website, feeds the content to an LLM via **OpenRouter**, and produces a concise, colourful PDF brochure.

---

## Prerequisites

- Python 3.10+ (Conda environment recommended)
- An [OpenRouter](https://openrouter.ai) API key

---

## Setup

**1. Clone the repo and enter the directory**

```bash
git clone <repo-url>
cd ai_engineering
```

**2. Create and activate a Conda environment** *(or use any virtual environment)*

```bash
conda create -n llms python=3.11 -y
conda activate llms
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your API key**

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-...
```

---

## Generate a Brochure

Open and run **`week1/day5.ipynb`** in Jupyter or VS Code.

Run the cells in order, then call:

```python
create_brochure_generate_pdf("Company Name", "https://company-website.com")
```

The brochure will be saved as a PDF to:

```
brochures/<CompanyName>.pdf
```

---

## Project Structure

```
ai_engineering/
├── week1/
│   ├── day5.ipynb      # Main brochure notebook
│   └── scraper.py      # Website scraper (fetch links & content)
├── brochures/          # Generated PDFs (git-ignored)
├── .env                # API keys (git-ignored)
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Scrape** — `scraper.py` fetches the landing page and extracts links in a single HTTP request.
2. **Filter** — The LLM selects only brochure-relevant links (About, Careers, etc.).
3. **Summarise** — Page content from all relevant links is compiled into a prompt.
4. **Generate** — The LLM writes a 300–400 word markdown brochure.
5. **Export** — `fpdf2` renders the markdown into a styled, colourful PDF with a branded header banner, coloured section headings, and a footer.
