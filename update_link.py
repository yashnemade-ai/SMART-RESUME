import requests
import re

# --- कॉन्फ़िगरेशन (यहाँ अपनी ज़रूरत के हिसाब से बदलें) ---
M3U_URL = "https://raw.githubusercontent.com/Sflex0719/ZioGarmTara/main/ZioGarmTara.m3u"
NPOINT_URL = "https://api.npoint.io/85fbd5b6462f28618069"
CHANNEL_NAME = "Star Plus"  # उस चैनल का नाम लिखें जिसका लिंक आप बदलना चाहते हैं
# -------------------------------------------------------

def get_link_from_m3u():
    try:
        response = requests.get(M3U_URL)
        content = response.text
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            if CHANNEL_NAME in line:
                # चैनल के नाम के ठीक नीचे वाली लाइन लिंक होती है
                return lines[i+1].strip()
        return None
    except Exception as e:
        print(f"Error fetching M3U: {e}")
        return None

def update_npoint(new_link):
    try:
        # 1. पुराना डेटा डाउनलोड करें
        current_data = requests.get(NPOINT_URL).json()
        
        # 2. डेटा अपडेट करें 
        # (ध्यान दें: 'url' की जगह वो नाम लिखें जो आपके JSON में है)
        current_data['url'] = new_link 
        
        # 3. वापस npoint पर अपलोड करें
        response = requests.post(NPOINT_URL, json=current_data)
        
        if response.status_code == 200:
            print(f"Success! {CHANNEL_NAME} updated with new link.")
        else:
            print(f"Failed to update npoint. Status: {response.status_code}")
            
    except Exception as e:
        print(f"Error updating npoint: {e}")

if __name__ == "__main__":
    link = get_link_from_m3u()
    if link:
        print(f"Found new link: {link}")
        update_npoint(link)
    else:
        print("Channel not found in M3U file.")
