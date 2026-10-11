# Deployment checklist (hackathon)

- [ ] Sign in to GitHub and create a public repository.
- [ ] Upload `app.py`, `model.joblib`, `reference_data.csv`, `requirements.txt`, all three evaluation files, `.streamlit/config.toml`, and `README.md`.
- [ ] Confirm GitHub shows `app.py` in the **repository root**.
- [ ] Open https://share.streamlit.io/ and deploy `app.py`.
- [ ] Wait for packages and model to load; run a sample prediction.
- [ ] Verify charts and leave-one-device-out table render.
- [ ] Save the public URL and optionally create a QR code for the final slide.
- [ ] Test on your phone and laptop.
- [ ] Keep a local copy and run `python -m streamlit run app.py` as a backup.
- [ ] State honestly: Streamlit-hosted research prototype, not Alibaba Cloud-hosted or clinically validated.
