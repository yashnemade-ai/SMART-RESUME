import requests

# --- सेटिंग्स ---
M3U_URL = "https://raw.githubusercontent.com/Sflex0719/ZioGarmTara/main/ZioGarmTara.m3u"
NPOINT_URL = "https://api.npoint.io/c0aa4f57f93105ce45de"

def get_m3u_links():
    """M3U से सभी चैनल्स के नाम और लिंक निकालता है"""
    try:
        response = requests.get(M3U_URL)
        lines = response.text.splitlines()
        m3u_map = {}
        for i, line in enumerate(lines):
            if line.startswith("#EXTINF"):
                name = line.split(",")[-1].strip() # नाम निकालता है
                link = lines[i+1].strip() # अगली लाइन का लिंक निकालता है
                m3u_map[name] = link
        return m3u_map
    except:
        return {}

def update_process():
    # 1. ताज़ा लिंक्स लोड करें
    m3u_data = get_m3u_links()
    
    # 2. npoint से अपना JSON मंगवाएं
    response = requests.get(NPOINT_URL)
    json_list = response.json()

    updated = False
    
    # 3. लिस्ट में हर चैनल चेक करें
    for channel in json_list:
        current_url = channel.get("url", "")
        
        # अगर लिंक के आखिर में # लगा है
        if current_url.endswith("#"):
            channel_name = channel.get("name") # JSON में जो नाम है
            
            # M3U में उस नाम का लिंक ढूंढें
            # ध्यान दें: JSON का "name" और M3U का "name" एक जैसा होना चाहिए
            if channel_name in m3u_data:
                new_link = m3u_data[channel_name]
                
                # नया लिंक डालो और आखिर में फिर से # लगा दो (अगली बार के लिए)
                channel["url"] = new_link + "#"
                updated = True
                print(f"Updated: {channel_name}")
            else:
                # अगर नाम थोड़ा अलग है (जैसे M3U में 'Star Plus' और JSON में 'Star Plus HD')
                # तो यहाँ मैन्युअल चेक भी डाल सकते हैं
                print(f"Not found in M3U: {channel_name}")

    # 4. अगर बदलाव हुए हैं तो सेव करें
    if updated:
        requests.post(NPOINT_URL, json=json_list)
        print("npoint.io successfully updated!")
    else:
        print("No links found with # marker.")

if __name__ == "__main__":
    update_process()
