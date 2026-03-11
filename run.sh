afconvert ~/Desktop/sample_speech.m4a ~/Desktop/sample_speech.wav -f WAVE -d LEI16@44100

python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the LPC analysis
python main.py                        # uses my_recording.wav
python main.py --wav your_file.wav    # use a different recording