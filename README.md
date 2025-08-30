<h1 align="center">🚀 Movie Recommendation System</h1>

<p align="center">
  <i>Personalized movie recommendations using collaborative filtering, built for AI/ML enthusiasts and learners.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Frontend-Bootstrap%205-blue" />
  <img src="https://img.shields.io/badge/Backend-Python%20%7C%20Flask-brightgreen" />
  <img src="https://img.shields.io/badge/Data-MovieLens%20CSV-yellowgreen" />
  <img src="https://img.shields.io/badge/ML-Collaborative%20Filtering-orange" />
</p>

## ✨ Features

- ✅ User-based and Item-based collaborative filtering recommendations
- ✅ Optional genre filtering for personalized results
- ✅ Most popular movies display
- ✅ Responsive and modern Bootstrap 5 UI
- ✅ RMSE evaluation module for model testing

## 🏷️ Tech Stack

| Layer              | Tech Stack                                      |
| ------------------ | ------------------------------------------------|
| **Frontend**       | HTML, Bootstrap 5, Jinja2 Templates             |
| **Backend**        | Python, Flask                                   |
| **Data**           | MovieLens 100k dataset                          |
| **ML/Algorithms**  | User-based & Item-based Collaborative Filtering |
| **Deployment**     | Localhost / Any Flask-compatible server         |

## 🔗 Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/fafd7001-2dea-48e0-b57c-219fb840bcc1" alt="Home_1" width="80%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/903af4d2-da60-4f96-b8fe-578d00b9fdb0" alt="Home_2" width="80%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/eea3e0e5-616a-4b56-a5d4-b139143cede1" alt="Recommendations" width="80%" />
</p>

## 🛠️ Installation

Clone the project and run locally:

```bash
git clone https://github.com/sharadj19/movie-recommender.git
cd movie-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
````

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## 📁 Folder Structure

```plaintext
movie-recommender/
│
├── recommendation_engine/
│   ├── collaborative.py       # User & Item-based CF
│   ├── data_loader.py         # Load MovieLens CSVs
│   ├── evaluation.py          # Train-test split & RMSE
│
├── templates/
│   ├── index.html             # Home page & recommendation form
│   └── recommendations.html   # Recommendations display
│
├── ml-latest-small/           # MovieLens CSV dataset
├── app.py                     # Flask app entry point
└── README.md
```

## 👨‍💻 Author

<table>
<tr>
  <td align="center">
    <a href="https://sharad.is-a.dev/">
      <img src="https://avatars.githubusercontent.com/u/85397332?v=4" width="100px;" alt="Sharad's Avatar"/>
      <br />
      <sub><b>Sharad Chandel</b></sub>
    </a>
  </td>
</tr>
</table>

📫 [Email](mailto:sharadchandel2005@email.com)
🔗 [LinkedIn](https://www.linkedin.com/in/sharadchandel2005/)
🌐 [Portfolio](https://sharad.is-a.dev/)

## 📝 License

🧾 This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](./LICENSE) file for full legal details.

## 🤝 Contribution

Pull requests are welcome!
For major changes, open an issue first to discuss your idea.

<p align="center">
  🚧 Built with Python, data, and a love for movies.
</p>