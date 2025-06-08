import os
import json
import pandas as pd

from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

### Setup ###
# Setup API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# load config ban đầu của LLM
with open('./config.json', 'r') as f:
    config = json.load(f)
    functions = config.get('function', 'giới thiệu nhà hàng')
    initial_bot_message = config.get('initial_bot_message', 'Xin chào! Bạn cần hỗ trợ gì?')
    
# load nội dung menu
menu_df = pd.read_csv('./menu.csv', index_col=[0])
# Tạo LLM
model = genai.GenerativeModel("gemini-1.5-flash",
                              system_instruction=f"""
                              Bạn tên là MienLuon, một trợ lý AI có nhiệm vụ hỗ trợ giải đáp thông tin cho khách hàng đến nhà hàng PhoBo Cuisine.
                              
                              Các chức năng mà bạn hỗ trợ gồm:
                              1. Giới thiệu nhà hàng PhoBo Cuisine: là một nhà hàng thành lập bởi doanh nghiệp Việt Nam, ở địa chỉ 22 Jump Streets, New York, USA 
                              2. Giới thiệu menu nhà hàng gồm các món: {', '.join(menu_df['name'].to_list())}
                              Đối với các câu hỏi ngoài chức năng mà bạn hỗ trợ, trả lời bằng 'Tôi đang không hỗ trợ chức năng này. Xin liên hệ nhân viên nhà hàng để biết thêm thông tin.
                              
                              Hãy có thái độ thân thiện và lịch sự khi nói chuyện với khách hàng, vì khách hàng là thượng đế.
                              """)


### Hàm trò chuyện của chatbot ### 
def restaurant_chatbot():
    st.title("Restaurant Assistant Chatbot")
    st.write("Xin chào! Tôi là trợ lý online của nhà hàng PhoBo Cuisine. Bạn cần hỗ trợ gì?")
    st.write("(Bạn có thể hỏi tôi về thời gian mở cửa, menu món ăn, ...)")
    
    # Nếu chưa có lịch sử trò chuyện 
    if 'conversation_log' not in st.session_state:
        st.session_state.conversation_log = [
            {"role": "assistant", "content": initial_bot_message}
        ]
    
    # Nếu đã có lịch sử trò chuyện, hiển thị lịch sử ra màn hình
    for message in st.session_state.conversation_log:
        if message["role"] != 'system':
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
    # Khi người dùng nhập prompt
    if prompt := st.chat_input("Nhập yêu cầu của bạn tại đây..."):
        # Hiển thị prompt của người dùng
        with st.chat_message("user"):
            st.write(prompt)
        # và thêm vào log
        st.session_state.conversation_log.append({"role": "user", "content": prompt})
        
        ## LLM tạo câu trả lời
        response = model.generate_content(prompt)
        bot_reply = response.text
        
        # Kiểm tra xem prompt có đề cập đến menu không
        if 'menu' in prompt.lower() or 'món' in prompt.lower() or 'thực đơn' in prompt.lower(): 
            bot_reply = ""
            for index, row in menu_df.iterrows(): # duyệt từng dòng trong DF
                name = row['name']                  # lấy tên món ăn
                description = row['description']    # lấy mô tả món ăn
                line = f"**{name}**: {description}" # tạo dòng mô tả định dạng đẹp
                bot_reply += line + "\n\n"          # ghép từng dòng vào chuỗi kết quả => cách nhau 2 dòng
        else: 
            response = model.generate_content(prompt)
            bot_reply = response.text
        
        # Hiển thị câu trả lời từ LLM
        with st.chat_message("assistant"):
            st.write(bot_reply)
        # và thêm vào log
        st.session_state.conversation_log.append({"role": 'assistant', 'content': bot_reply})
                
                

### Chạy chương trình ###
if __name__ == "__main__":
    restaurant_chatbot()