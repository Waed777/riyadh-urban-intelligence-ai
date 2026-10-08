# 🇸🇦 Riyadh Urban Intelligence AI

**Open-data geospatial intelligence for Riyadh — built as an end-to-end AI/data science portfolio project.**

> **RCRC Open Data → Geospatial Features → AI Accessibility Score → ML Zone Segmentation → Interactive Map**

## Live Demo

**Streamlit:** https://riyadh-urban-intelligence-ai.streamlit.app/

> If the demo is not deployed yet, run the included Colab notebook first. The repository is designed so the Streamlit app can be deployed directly from GitHub.

## What this project does

Riyadh Urban Intelligence AI converts public Riyadh mobility data into an interactive location-intelligence prototype. It evaluates candidate grid locations using proximity and density signals around metro stations and, when available, additional RCRC mobility datasets.

### Pipeline

1. **Data acquisition** — RCRC Open Data API v2.1
2. **Geospatial feature engineering** — Haversine distance and radius-based counts
3. **AI scoring** — normalized accessibility/connectivity score
4. **ML segmentation** — K-Means clustering into four zone types
5. **Explainability** — Random Forest surrogate feature importance
6. **Visualization** — interactive Plotly/Folium maps
7. **Export** — CSV outputs for downstream analytics

## Why it matters

The goal is not to claim that the model predicts commercial success. There is no ground-truth business-outcome dataset in this prototype. Instead, it demonstrates how open geospatial data can be transformed into a **decision-support layer** that can later be validated against real outcomes.

## Data source

Riyadh Open Data Portal (RCRC):
https://opendata.rcrc.gov.sa/

The project uses the RCRC Explore API v2.1:
https://opendata.rcrc.gov.sa/api-console/explore/v2.1/

Primary confirmed dataset:
- Metro stations in Riyadh by metro line and station type — technical ID: `metro-stations-in-riyadh-by-metro-line-and-station-type-2024`

The portal currently lists Riyadh metro, bus, roads/traffic and other urban datasets, making it possible to extend the feature layer over time.

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- Plotly
- Folium
- REST APIs
- Geospatial feature engineering
- K-Means clustering
- Random Forest explainability
- Google Colab
- Streamlit (deployment target)

## Run in Google Colab

Open:
`notebooks/riyadh_urban_intelligence_ai_colab.ipynb`

Run the single code cell. It installs dependencies, pulls the RCRC data, computes the features, generates the score, creates the maps, and exports CSV results.

## Repository structure

```text
riyadh-urban-intelligence-ai/
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── riyadh_urban_intelligence_ai_colab.ipynb
├── src/
│   └── README.md
├── data/
│   └── README.md
├── docs/
│   └── deployment.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Responsible AI / limitations

- This is a **prototype location-intelligence ranking**, not a claim of business-success prediction.
- The score weights are explicit and should be validated with domain experts and real outcomes before operational use.
- Public datasets can change; the notebook uses the RCRC API rather than embedding proprietary data.
- Any commercial deployment should include validation, monitoring, data-quality checks and bias/coverage analysis.

## Roadmap

- [ ] Add validated bus-station and traffic-intersection layers through dynamic RCRC catalog discovery
- [ ] Add land-use and commercial-services features
- [ ] Add temporal mobility signals
- [ ] Add model calibration against real business/location outcomes
- [ ] Add an LLM/RAG assistant that explains zone-level signals using the underlying data
- [ ] Deploy production API + dashboard

## Author

**Waed Amari** — Data Science & AI | Machine Learning | GenAI | Analytics

GitHub: https://github.com/Waed777
LinkedIn: https://www.linkedin.com/in/amari-waed-825a8b326/

## License

MIT — see `LICENSE`.
