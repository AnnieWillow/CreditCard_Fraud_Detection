import streamlit as st
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import math
import joblib

# Page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Custom CSS for Sidebar Buttons
st.markdown(
    """
    <style>
        .stButton > button {
            width: 100%;
            background-color: #004466;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px;
            margin-bottom: 10px;
            border: none;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #0077b6;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Embed the logo inside a div for centering
st.sidebar.markdown("""
    <div style="text-align: center;margin-bottom: 20px; margin-top:0px ">
        <img src="https://cdn-icons-png.flaticon.com/512/6963/6963703.png" width="150" style="display: block; margin-left: auto; margin-right: auto;" />
    </div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("<h3 style='font-size: 30px;'>Credit Card Fraud Detection</h3>", unsafe_allow_html=True)
if st.sidebar.button("📊 Data Overview"):
    st.session_state.page = "overview"
if st.sidebar.button("📈 Fraud Analysis"):
    st.session_state.page = "analysis"
if st.sidebar.button("👤 Customer Analysis"):
    st.session_state.page = "customer"
if st.sidebar.button("⏳ Date/Time Analysis"):
    st.session_state.page = "datetime"
if st.sidebar.button("⏳ Model Prediction"):
    st.session_state.page = "model"


# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "overview"

# Function to show Data Overview page
def show_data_overview():
    st.title("📊 Data Overview")

    # Display an image
    st.image(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAATcAAACiCAMAAAATIHpEAAACAVBMVEVPw9AON1U8RE/////5zFdPw9FGoq1PxM7/3qtQwtNGwtGW2+BJu8w9RE7/wQ4ON1MALk55iZgwOkejpqptzdZ9ztXy8vKo3eQAIEbAydDxrBNaYWqQk5qRnagAK088RFFxeouUwcbi4uK9w8oAHT/O0NX/3qdhc4Jxeow5Rk//3K9zfopFn61ixtKbECg9QExttcP/6LP9zFMAADI+oqnAAACaLjxNi5D5//wyN0g8O0S6vL9HVGQ/vMfx/P3V19giMUT1thClPkz/6K0zSlP/6rw0wOZatsBEbHkAGkcAJEEAKE9ykZ+9pKdsorKgBRKcCB3Pnp221d6NfYM6XWpCd39sn6Y1Lz+bEiK6ABWmKzmIABBEzdnE6O/48Oq3S03qu7y3HDHJd3vq1NGpf4Rdv9nU7vG2U163YmqWanWfW2RyvbKaupqcYWb9sDP/qBz8uFnu8eXaqlS3vW3bs0n21I2Ev5z51HzkwjP28dSewIbvxB5WwMBrwaeAbHXOxE2erbbNxWp1emuWkX66sJXTxZ63rJNvcWVQU1TQlJnCNUPcXGnc2I/VcnyWR13OtWRGU3GUv43bsC/o1rOpw3Pu02W9xWBhP0e2tGPU6Nac0bwArd2sx5IkLjaC0+ldWT2emDh8dEPIwCzz5CeTikM9Rj7Eti6qqTdXVWZMZXy1tMrvHbw7AAAW9klEQVR4nO2djV8T55bHx4QnDzNmeBkkkAthJCHYMcThRRsySQgt4IC62tpgtDdae11p1XZb16q1pVsUUdrd3q2r3rvblt7b9rZb96/cc56ZhLxNkg4xUMzvo+SFZJj55pzznHOeZyYc97sT2ddNyv9C2CtYvUnY66a89TaFnkT5be4iNbnZU5ObPTW52VOTmz01udlTk5s9NbnZU5ObPTW52VOTmz01udlTk5s9NbnZU5ObPTW52VOTmz01udlTk5s9NbnZU5ObPTW52VOTmz01udlTk5s9NbnZU5ObPTW52VOTmz1ZcqMn/umk1Zua3Ky5jZ7yvha2eFNlbuTF5nbWdyhscfhVuQkVrHF3qIKfHjmZsXhTFW6ZHrpz7I0QY08p4ev5YZZwI/AHqAAHHj5k5aZVuNETr4cot1PIUfwMCU/gDldHcOXtjRJCbHMLnzp9aBS2UJf927J4OJQMOA5YG63jZku4UZ7CP57at7fwoTcOuXcIN9gJPTKfmk0tHq7vHpVyI1Q/g0+FLYeFan46+tqZzLb7KXz6BEKtKxXVVEVUVS09v5xhT9ZFpdw4or9KYPuGvZW17QrceAiP3Muj9dm5LQiCNJrAfFp1GJIdanoxzNfLV8vEN6IfOSNwfPhsGAajssZtzY1gGKY7gBvsiUBcUVExsTkUWVTUWUBZp82Xs7dzr7L4dlmgoDJvsuRGMIjQncANP8CIpKqybHITRVlWxaj+PP30zZNuTnD/sbunZ5+7HJ/y3CCc8J0wqmy7vUEUg+GNRDTRUSBRdIhRF+XqEXsLuJluqb+ccbu7z19wgs4n+FLbtuAG4N4Mo8ltu71BkM2MpIuwobOKqqTzlWrrWlXADd2SEsH9Ws/SpJNp0tkj1MwNxqpzb2QIcqtnqvSbhWOC4ErLJdhkGcHRegwOhX5KBN69vHfqrYvOnCYTpUWTNbe3X+PB4F4e3eYshKd6VCnlBoOqAoNDuD7ccm02QhLdPVPA6sKpP13Kgev+LfZ2dlSHgnCb/RQ+ORpXy2BDi3Mo2iIhWy2gCd3XbVDh4UPqnmKuCT76+kXTTSHCCSV2XZ4bjr2HIHgI9J89y3xdvMGmCBdRy1FjUmXJhdmSfXIYwvnublbuYm3V7SynqUSN3DA/P3uZJvYtOT09btt7VQfxepkxYZMbpHFbLFaJMPrOuyfZJniaWHJOZmH5g/5sfHMulzRgrPI3mrlyZd95Dyos1GPUsidCU9bY0Fm1CNmaN5DMVZ/vKtsE4RJTpmc6PTG/xx/zmOR6hGKTLstNAPHknbcuTiG3pcRW9msLgrybuDRH2eiGgroBsrjw1hpdRDjz5psnzPake8rE5ow5k10e+GloqaTrXY6bcO29997bO+W8dJHZ22S3wPPb0b6EP5mZFS2xGRIXt9ocyfS/z7BBrAM/NZ3U45fi0qQnaD4OF09slXDjeeHav3zw4dwFj6nrnuv/eqMuCeZvFHYkXGlRViphk0XNbvzN4iYnrmQoZ7R595qcYs4gcDMNDiywZEKwhBu8+aMP5zw5+cHgbrbd2gaDA26ZFFTzFQMceOui7fhmWuqJKyw9I5iJmNyCHmcoHj+Qs7eSWFXqp8J7rdc3qfn9YHi32z6mjTc4OBgdolu5pDfPTWVZsrtnhOisHj3xvhH2IZ4mstVVzNkXB7EhIgj0KuYh2L5PdBdQA03daWtr+0TgtpIn2RLlF61zt5yjytpI2U5PVREidOoCFeiJ93MbEHqyAS4mxWXR7/R4Y0GPJ+ns5godrpAbTYSXPEXY/FNPgduny/jJNDb/JWGpio86WL2VsucLPJcJXclA9nfiSi4F5JdzyW6wy+uPxZLsrh+bIhW4CT2TxdT8fjS3tra/8MZ8UuNEOUhCqktRJdt/IHREh8PqvJKbJCYsgzMtLsAS3yTcTmLJUPDewvhGY5PF1Pz+Fsat7TO+wYMq5ebF6n4KRYMWsdn75UNf/huhwon3zy+bQwQ1HTUYC3iM26AnGfPG/M7uwpGx0N70rmQxNf9jA1vb54lGJyNhqQibKIoQz+TigUJN2Qq9YAehNzIchCYoCBKEpc883YeWFvSDqXkCgaDHBBhznncXfDR53AgnDHZ1FWPLmltb263SrsDzlZ6umLpt4rTnqJDNh94Qwkaq66bMKqBEnXLGkrFgEkIbQGN5iMcbYI5qZW+UrnR1BQupeeay2No+5hpsbxFNKRgXypiaofRhW90k8NCrS2Zh1SNwkIxApp047wwGA2BwwQAbE/yBgP/CxT9dcO4rsJocN9ZV7+pK+guxTY7nuLU9abC9pfJrLFmSotFpQ5IkFQDUIjbSciIk9i29nW3rTt5tuYuHZ7SSkl4AB7VCMuD1X3j9nXPnjryO3fK8d+dzI6OxQh8Flx3bxIa5SN2YVDsoSJcgC1EVAxmgWr+3+uD+/TXU/Qerqw/XJcAnG2TVVM3ccI6O58G2hEQ32NqFt05dMEqpuUu3b1GsptzoqEFwU7A6z6UvvjyHeueScykvwIFP73UvMzMCE6XfO/ODGytO88wNcpHLlGuYzREIb4ZVKav3H4+BhodbD7YOt7YOD8ODmbm1B/ck0+Bma90mW8tEecG93G16qIf9nLwzd7Ol5S7wpO4lZ8yT9HqTAO3NL//97XNvf3WRsd1cK4IUTrs3V7hduwPFaB41j2emrUCfCVxjcl/0mJG0g/WQpIdzAAuQFergcOuYaAY8Sa9xswTqA2H51pP/uJ7Xzr0+d7ulZXx8vGX8GkAV9sJAkLxw8atTR7786q2zbJohyWr73JFD5fznL/44ymp9nDL9zw/vXLjuMaCx1pvn00JunwtCg3JfsIyIarihtDp2cOz+2szwJjmwuPtrB1vH1hVjxNVcte0WLyx/9t3dj2/fybbZgNlNhqwF/rWMfyOgwzsnL73FvPPNLwDaJAytLAFeyjZeCM+FT3W/etKoasH4nkAMe+XTTz+GbTy9ffvmzZnbf3glJwbuUb0Ws1QRpKGZeeCGUKKrra1jq9MP78+1mo46NvdgvX3t4PDYQ8PeZDVSAzfwz1t3ARJSyzJ7ynhlNf4doMAm3LtHIKS9Dt7p8ceS2Xb51GbDihfeOHfVCFnwBkKLzOuVPxTrieXC9PoKW5YpM+uVHoBHrsa/VpQHc2NI7fGqHI/H1+DeQ/Ml6mJ1briIEqjdvJNldpvZWb7Gb+DhQclw6V0MaZi/eTbdeWlztpDymVdHN1tE/KNCbkXQPvjomsA3rLan2aQXubU+mAVUs8rq4+G1h0gtPrs2fDDLTVbnM9W5Xabup4za9Tszt1uKmTFudxENCaMTQ3mazIuB8FR3blM8V8iNCp9XMLf/ugVjLt+YVYRY9GSLLAn8dPgxgxWX1mZWv47H++K/rs+1ts6tm9zEVHVuAqXdMBoAs007K2b3lJmtG0sIj8dZoMm9m3MMYDrALc+E+M+sze0RxeUldV1ZW0EQO7JJhnRv7ODw8AMFsEVX54bnHiJA+TEMqHPr5qSNmKohvBFsgs+Y0MqZGzy7jK8ER51kyUmej/aEcyPiKNoZcsvzPP4vFub238uCUTw3KA/hqZ7lJq8Dt9aZtXt999bGhoHWg4d9q4/hXuucZIwcomO2hk+TJADAzbK8ctyu4SFibZ9vaUt7lxO5iUCS+TMuLANuebUdzy2XNbePrtV3KXJV4RLbaLaMmmb5W+vMGI4KkIO0js2M4dA6vDa9ya3q7kG2ex7trSK3R4y/e2oyDxqXEPjcvDOM86fOgNcVcOONXCTP3D4wApvANXheBuzNneUmTsPQOdwKlmZQY/9bDx4cnlmVjDqrJnsjHB+G8FaZ2xO0Dz4TNrqXUz373FCVQiDLNdJhKD37cpgr5AbvofTjfHNDeJ9QyAYbPp0F8U02rAkTXywPhvMrBkQ3tu4wKjFZqSG+wcELS847lbDBgIp2C2xozxKEtISA2QsGdZL7XCg9e+WkmYdkcCUvPomLGx8VRrcbFOswvsFzMjgExRVcVolYNuZai4Vm93jDbDPJ2mJNu0e6ndercDOLdyj7EyVrG4w9y7yNy7RxXBiJpzbXyhIzF8kGNqFBmW7J/nGLOAnIsgzpQQk3NLd7UbObJGqu6hvE/Ckx5azIreVpdokgtTyVCbgBNv5qPKqJYirnh1T4LGduENi2iRoWL3rUmKsXHfLPpQYH5jYtG4W/Q56txdwIds2ct6sMqDnHsppczFw9wbvmFYeiKKqcXs4NGDzLRXBI+GSZFxqVrxUL0p3MotnukGWMcMPF5vZQxukG/H16pNZhKzFVcUAdz3HjueItUtb6IELmxNt/TauyA6jJihbJYSZ0+dM2I2PDwNbgOdM8ESqZ5yzI4sZaPjYs72ceTDOoiiiqqVpXXVKhZ648t2xH5JbFIkmCQyNPM3okPvvXQ68q4AmyAuBkmiXMU8hFMLBt++lFvCttpmeA53GBvQ3PrE2bvV5FlHSh1gX5PK2YiIyPPxHKGwp2iTPhSCqqgofOqj+LopSOSvi3efMzw4mdz29x244NE65I2jA3cNWfH4/ltSzH7k+bczaKmNZrbm7xnPA/luY2Pv70xq1lYnEaDKGHU5IKxs38U57uWxkc7E/L6mLGeAeP4w4PgW1b199THP7AYxajyIylI9L9seGsm86tTkNkM4Jf2gUWUuPOCrzwTXk7a3l649E1ozAo3BbbExp2zUtRVRUhX1QlUVTSg98fPfrt0b3rajRjxFbCTr7j6nImyhaEGSWe8hERHdl+0vTDNZxkmBmbu7+eWzgiqq4MmzWoLb7x3LWcgWV/ALO7j67hIlOLbVB9UdGyzUA11C/J2spRQ3s1ybXt50vmCdwCJ4r0SMps+jLTmhbvra6uPpSm8ydS5yNchudrcw+wCPo0S8z8f/e7azAC8hbkgbRrFvzTGNalvq5gMCSJv37/LZjbt2BxPm1+iwuM6y1CR+JpCCib699EnESVJEXBRUg5mKKmpVxcbfaGKdyNcdM1DWYU7QzqT6G8r0N6kcZ9kBWEFgv2923AXviOMmygQS1atxMTtyzsktOIrIlsrM+bs4e0SURbU/IWS2Mmpc26LM/ozhd4P//IGAOA2S1khgkrO5m+fD8WYgCk3w5Rivd3Jbv6NqT+LsmhnTnKoMGPwY3o4R3iqOigmRGl0hqu0uUOQC6DGUGVbjSk8ddw1um7byCnF2pYjgMBICXKYiiWjIWmN/q6krF+0aH2G9CQW1Sc3yGXvOBJRo9rag1LuPKlpOcprWFkJcvf3cI6SKjNuSDMujSpKxaSNjC29YuShJ3U7w1qR4/2S0p0h3ATaIStIq++1jJfmP+6ajnnnvBGd6jGg4V0NupwbKx3BSC2sXUpsqhEfZe/ZSZ3Og35XGRncAunNIhqv9Hc2Fkg6UWuOg42UtecnmJDa16VMLZNT0dlhxFu5Q0f5G9Hvx/UINyKqa0e8dYFu6nLKkszqpzvUUaQk6awvKzFlKoFwtzrgO+IJvdp0x1/+/vfzTM6YSiS1n0rK31pGTun0e0fTwGbpIl2pYhqfJmr9/JtQiVZUX748acff5TMBg1L5qJRqB8csqbN74BLSOmQfEi2hc2d+k/w0kWoSv/x4y8//fRD1MwbJcwmIU1SNWl+ZPvNjdMlUYUUya5CaXTVeoMjuhaVfwZz++mXPaxSlvr6cYBQtWhqhGa2u17A6ZBZ1dHnX+m0q0BSk7V5q2uP2d8zGHOmf/n0px//EXWIG0ooEOwCaunZiE45vuGnwxQL0qBF1SEl40MdNjWw0BlIK9GROh8I5SOqEv3hf/82EJUkyIC7QmpUUyI62f72BxMPwU0OrQzt2dO+x6YmgpDcK/U+Flz6KUYHJCXU7+/qX4+mpUUXdn93QGBD0UVVlPp9HUit3Zb2DHn7oNAfqe9+QdkHtZbc158MALRodN4VZjPRDVoLWF0wygO3AbvGBgJuiqzOk7r2KIBbRFNCyf71jY1oKhLGgaDO16DbmiBP2iK3Xm8fJAez9V2VgQNWVFbUaDoe0TONPvulqogLsqI6cHPIUk1NpZqFFwKZ1zQZhk/K11zYNkwkom3ZTxk3UdPrWjLgIg/qcmV4HvvKO+d6n6YIZCGiFTeI+fi/Y2Cgo4M9qMDNobnq7Kic2WipfSajgWILoS25IbaOhbhvpbN9wTpRMbipkR3nTM9PhOIUeAU/bR+IG0vivR0dlbmJ8y8ONhBO7lXg1vEsuwjSM91RmdvsTklJGyEdhgW5ErfNdcrJBQuDM7lFd1rsfp4aqcxtYCVvrfIBC4MzuMnpel378vegRU0Wrf20fcifx23leEVumusF4WaeV1SJW69n0hnwAjM8eTu4UJnbDpkqee5CbkpFbnsWPM7QwoTP2bcwEXMmeytye2EGVMjveXaehzW3IY/z2J49cefEno4DzuBQJW6iI0V2QM+/ASI4ISNX4+abmFhxhiYmvM5ARXuDzVhekH13ieC0eGV76whB/uGfeobXypqcLf+aHLd0jSdA/+6FveiK3NoHFr7uSjqDnX5nwNfeawE3y62Whfm7QoTMV+TW3h5YWVgwrpsYWugIrByrwE0WX5wBNTMrVuJ23Od09h2P9x2IPzuwZwEePOso1xUxuSna4nYfUKNEoxW59UK1kAz1Lgx0DEy0+zxgdOVfZvipoqZejAKVsIl6a27Zmj5w4FjIvHbiULlmUjYPER0vBLfccGod3yALCeT6ISsrzuSQtZ+KtV9R5Hcuwkc0tgrJ2k+9E+3Zyr5/SEqCn1pyU/Gkhu0+pIaIp3parMCtow+vyIMdpC52M4V+Wg6vMS7IYp2nZnaueJfkqGBv7YjL45MXjk/EV/Bs5GDZiiGbv83qL0R845jFKfi9MRXykM6JAWNuBgp7Z7xsr5zNnypaaocsuW2ICD+rsXUO5bhBLT/Zno1oA/kPSrgp+CVRO+acgucvytFUWrHgtqejHZ9vL/OgiJuYXsQLVbw4BkcpH15MV553brd8kOMWjRhXad/uw2mYCPuirDxubIWRBbxCgNkXtvd6fx3JsCuNvTh+iqL8oIWf1qL2PQsBnfI7beFLA0T5vT6LKZeq0NDoegP7XjRTY+Jpd4CttxwYOAaq0fQGQOy1He1JPEdr94Mj7EsIqfn1LiDqPr1ybKh3wreCerbQm6ehrPD+5tMLcfZa31DvQrt3kBoXsCA5bevxPTfljg1/IETOfToZCASSyWQQFKhFQXxpMukNeIODicuIjX0c3C6em6EGOLx167ruGhk5eebkmUDQGwx6QcFNFTwoktdrvNh75uSoCwRbSuCXQu7iVSK4DC+hH3YddjEddg32h0D7Q6UK5JFa2Xx6/0svmffwzsqoKyddd9d0rtbvTeyaiDo7xBGXeTMy+PUALg7sPT7Qy05JyJ2bMPDswKa+Hig8c+E4RLxjeNuZx41tVed2GTtc7+l2FenwyOBLODAe8x2IrxSNpvmcigbUjk6fz4drbY77Ros36dplU1uQLiRGwDFHCg9yNIYNomOxzmfe2hPgDl/XygoAHxgKllCDv7Hriq48F81Z3OlOPM9ooXdgKN9NK6q9A9KSoeMdHUP/N3i41N70XcWNrcvWS49y5LT3JRgW9u8PvbS/VC+hsg9Chb/Y3xkcPDxSssGy38/8+xaW8nopu0G7KjU1924ytQKxA9Ozw+ph5rhwm6d8Zx5B5d0aLzTTGPaU+TykcBz7SvTtPr7nIoLnnxCjVoBUjuW+CMPgYiKoqCxE49FhTHkTLPNgXyqxS1NfYn6BN81WlSY/LFoTCd1QRWzGSxLAqrg8oMaJLrvWT8vrt39dzPPYi6aaaqqppppqqqmmmmqqqaaaaqqppppqajfo/wEsHxWEhU6fzwAAAABJRU5ErkJggg==",
        caption="Fraud Detection System", use_container_width=True)

    st.markdown("""
    <style>
        .upload-section {
            border: 2px dashed #4CAF50;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            background-color: #f9f9f9;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='upload-section'><h4>📂 Upload your dataset (.csv)</h4></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv('credit_card_fraud.csv')  # Use uploaded file instead of fixed path
        st.success("✅ File uploaded successfully!")
        st.write("### Preview of Dataset:")
        # st.dataframe(df.head(10))  # Show first 10 rows for preview of dataset
        st.dataframe(df)  # Show all data


def show_data_analysis():
    st.header("📈 Fraud Analysis")
    st.write("Show key insights, distributions, and statistics.")

    # Load your dataset (Ensure df is available)
    df = pd.read_csv("credit_card_fraud.csv")

    # Options for the selectbox
    options = [
        "Please Select An Option",
        "Cities with Highest Fraud Rate",
        "Cities with 100% Fraud Rate",
        "Fraud Level By Purchase Category"
    ]

    # Only show the dropdown when "Data Analysis" is selected
    option = st.selectbox(
        "📉 Choose an option",
        options  # Make sure options is passed correctly
    )

    # Fraud and number of transactions by state
    fraud_by_state = df.groupby('state')['is_fraud'].mean().reset_index()

    # Fraud and number of transactions by city
    fraud_by_city = df.groupby('city')['is_fraud'].mean().reset_index().sort_values(by='is_fraud',
                                                                                    ascending=False).head(20)

    # Isolate cities with 100% fraud rate
    full_fraud_cities = fraud_by_city[fraud_by_city['is_fraud'] == 1]['city']
    # full_fraud_cities = full_fraud_cities['city']

    # Fraud and number of transactions by category
    fraud_by_category = df.groupby('category')['is_fraud'].mean().reset_index().sort_values(by='is_fraud',
                                                                                            ascending=False)
    transactions_by_category = df.groupby('category').size()

    # Execute analysis based on user selection
    if option == "Cities with Highest Fraud Rate":
        st.subheader("Cities with Highest Fraud Rate")
        plt.figure(figsize=(6, 12))
        sns.barplot(data=fraud_by_city, y='city', x='is_fraud', palette='Blues', orient='h')
        st.pyplot(plt)

    elif option == "Cities with 100% Fraud Rate":
        st.subheader("Cities with 100% Fraud Rate")
        st.write(full_fraud_cities)  # Show cities with 100% fraud rate
        # User selects a city from the list with a default option "Select a city"
        city_selected = st.selectbox("Select a City with 100% Fraud Rate",
                                     ["Select a city"] + full_fraud_cities.tolist())

        # Ensure the user selects a city and display the records for that city
        if city_selected != "Select a city":
            st.subheader(f"Records for {city_selected} with 100% Fraud Rate")
            st.dataframe(df[df['city'] == city_selected])  # Display the table of transactions for the selected city
        else:
            st.write("Please select a city to view the records.")

    elif option == "Fraud Level By Purchase Category":
            st.subheader("Fraud Level By Purchase Category")
            plt.figure(figsize=(6, 12))
            sns.barplot(data=fraud_by_category, y='category', x='is_fraud', orient='h', palette='Blues', hue='is_fraud',
                        legend=False)
            plt.title('Fraud Level by Purchase Category')
            st.pyplot(plt)



# Haversine function to calculate distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of the earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def show_customer_analysis():
    df = pd.read_csv("credit_card_fraud.csv")  # Update with your dataset
    # Calculate the 'distance' column
    df['distance'] = df.apply(lambda row: haversine(row['lat'], row['long'], row['merch_lat'], row['merch_long']),
                              axis=1)
    # Create Unique Customer ID
    df['customer_id_test'] = df['dob'].astype(str) + df['city'].astype(str) + df['job'].astype(str)
    customers = df.groupby('customer_id_test')
    unique_customers = list(customers.groups.keys())
    df["customer_id"] = df["customer_id_test"].map(dict(zip(unique_customers, range(len(unique_customers)))))

    st.header("👤 Customer Analysis")
    st.write("Analyze fraud patterns based on customer attributes.")

    option = st.radio(
        "Select Analysis Type:",
        ["Please select an option", "Unique Customers",  "Customer Purchase Frequency"]
    )

    # Proceed only if the user selects an actual analysis type
    if option != "Please select an option":
        if option == "Unique Customers":
            st.subheader("🔹 Unique Customers")
            st.write("Identified unique customers based on `dob`, `city`, and `job`.")

            # Show first 5 rows
            st.dataframe(df.head(5))

            # Search by Customer ID
            customer_id_input = st.number_input("Enter Customer ID:", min_value=0, step=1)
            if st.button("Display"):
                filtered_df = df[df['customer_id'] == customer_id_input]
                if not filtered_df.empty:
                    st.subheader(f"Details for Customer ID: {customer_id_input}")
                    st.dataframe(filtered_df)
                else:
                    st.warning("No data found for this Customer ID.")

        if option == "Unique Customer Behavior":
            # Customer Behavior Analysis
            st.subheader("🔹 Unique Customer Behavior")

            # Reset customers object to calculate average purchase price and distance
            customers = df.groupby('customer_id').agg({
                'amt': ['mean', lambda x: x.quantile(0.90)],
                'distance': ['mean', lambda x: x.quantile(0.90)]
            }).reset_index()

            # Flatten column names after aggregation
            customers.columns = ['customer_id', 'avg_amt', '90th_amt', 'avg_dist', '90th_dist']

            # Display the result as in your notebook
            st.write("### Customer Data with Average and 90th Percentile Values:")
            st.dataframe(customers.head(5))

            # Merge with the original dataframe
            df = df.merge(customers, on='customer_id', how='left')

            # Add boolean columns to indicate if the purchase price and distance are above averages
            df['above_avg_amt'] = (df['amt'] > df['avg_amt']).astype(int)
            df['above_90_amt'] = (df['amt'] > df['90th_amt']).astype(int)
            df['above_avg_distance'] = (df['distance'] > df['avg_dist']).astype(int)
            df['above_90_distance'] = (df['distance'] > df['90th_dist']).astype(int)

            # Button for visualization
            if st.button("Show Customer Behavior Visualization"):
                # Visualization: Relationships between customer behavior and reported fraud
                columns = ['above_avg_amt', 'above_90_amt', 'above_avg_distance', 'above_90_distance']

                for c in columns:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=df, x='is_fraud', y=c, palette='Blues')
                    plt.title(f"Relationship between {c} and Fraud")
                    plt.tight_layout()
                    st.pyplot(plt)  # Display plot in Streamlit
                    plt.close()  # Close the plot to avoid overlapping

        if option == "Customer Purchase Frequency":
            # Ensure your date-time field is a datetime object
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

            # Sort transactions by time
            df = df.sort_values('trans_date_trans_time')

            # Create a 'date' column
            df['date'] = df['trans_date_trans_time'].dt.date

            # Set index to the transaction time
            df.set_index('trans_date_trans_time', inplace=True)

            # Create a dummy variable
            df['dummy'] = 1

            # Rolling count of purchases per hour
            df_counts = df.groupby('customer_id')['dummy'].rolling('1h').count().reset_index()

            # Rename column
            df_counts.rename(columns={'dummy': 'purchases_in_last_hour'}, inplace=True)

            # Calculate purchases per day
            df['purchases_today'] = df.groupby(['customer_id', 'date'])['dummy'].transform('count')

            # Merge rolling count with original data
            df.reset_index(inplace=True)
            df = pd.merge(df, df_counts, on=['customer_id', 'trans_date_trans_time'])

            # Remove unnecessary columns
            df.drop(columns=['dummy', 'date'], inplace=True)

            # Display processed data
            st.write("### Processed Data Sample:")
            st.dataframe(df.head(20))

            # # Visualization: Fraud relationships
            # st.write("### Fraud Relationship with Purchase Frequency")
            #
            # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            #
            # sns.barplot(data=df, x='is_fraud', y='purchases_today', palette='Blues', ax=axes[0])
            # axes[0].set_title("Purchases Per Day vs Fraud")
            #
            # sns.barplot(data=df, x='is_fraud', y='purchases_in_last_hour', palette='Blues', ax=axes[1])
            # axes[1].set_title("Purchases Per Hour vs Fraud")
            #
            # st.pyplot(fig)


def show_data_distribute():
    st.header("⏳ Date/Time Analysis")
    st.write("Examine fraud trends based on timestamps.")
    df = pd.read_csv("credit_card_fraud.csv")  # Update with your dataset

    # Ensure datetime column is properly formatted
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    # Extract date-time components
    df['year'] = df['trans_date_trans_time'].dt.year
    df['month'] = df['trans_date_trans_time'].dt.month
    df['day'] = df['trans_date_trans_time'].dt.day
    df['weekday'] = df['trans_date_trans_time'].dt.weekday
    df['hour'] = df['trans_date_trans_time'].dt.hour

    option = st.radio(
        "Select Analysis Type:",
        ["Please select an option", "Year", "Month", "Day", "WeekDay", "Hour"]
    )
# Mapping radio button selection to column names
    dt_mapping = {
        "Year": "year",
        "Month": "month",
        "Day": "day",
        "WeekDay": "weekday",
        "Hour": "hour"
    }

    if option in dt_mapping:
        dt_col = dt_mapping[option]

        # Plot transactions count
        st.subheader(f"Transactions by {option}")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=dt_col, palette='Blues', ax=ax)
        plt.title(f'Transactions by {option}')
        st.pyplot(fig)

        # Plot fraud rate
        st.subheader(f"Fraud Rate by {option}")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x=dt_col, y='is_fraud', palette='Blues', ax=ax)
        plt.title(f'Fraud Rate by {option}')
        st.pyplot(fig)


# Load models and encoders
isolation_forest = joblib.load("isolation_forest.pkl")
xgboost_model = joblib.load("xgboost_model.pkl")
label_encoders_if = joblib.load("label_encoders.pkl")
label_encoders_xgb = joblib.load("label_encoders_xgboost.pkl")


# Function to preprocess the data
def preprocess_data(df, label_encoders):
    df = df.drop(columns=['trans_num', 'merchant', 'is_fraud'], errors='ignore')
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['dob_year'] = df['dob'].dt.year
        df['dob_month'] = df['dob'].dt.month
        df['dob_day'] = df['dob'].dt.day
        df = df.drop(columns=['dob'])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df = df.drop(columns=['trans_date_trans_time'])
    categorical_cols = ['category', 'state', 'job', 'city']
    for col in categorical_cols:
        le = label_encoders.get(col, None)
        if le:
            df[col] = df[col].astype(str).where(df[col].isin(le.classes_), le.classes_[0])
            df[col] = le.transform(df[col])
    df = df.fillna(0)
    return df


# Predict using Isolation Forest
def predict_isolation_forest(df):
    df_processed = preprocess_data(df, label_encoders_if)
    predictions = isolation_forest.predict(df_processed)
    df['prediction'] = ["Fraud Transaction" if p == -1 else "Normal Transaction" for p in predictions]

    # Drop the 'is_fraud' column before saving
    df = df.drop(columns=['is_fraud'], errors='ignore')

    # Save the predictions to predictions.csv
    df.to_csv("predictions.csv", index=False)
    return df


# Predict using XGBoost
def predict_xgboost(df):
    df_processed = preprocess_data(df, label_encoders_xgb)
    predictions = xgboost_model.predict(df_processed)
    df['prediction'] = ["Fraud Transaction" if p > 0.5 else "Normal Transaction" for p in predictions]

    # Drop the 'is_fraud' column before saving
    df = df.drop(columns=['is_fraud'], errors='ignore')

    # Save the predictions to predictions_XGBoost.csv
    df.to_csv("predictions_XGBoost.csv", index=False)
    return df


# Streamlit UI
def show_data_model():
    st.title("Fraud Detection for Unseen Transactions")

    # File uploader
    uploaded_file = st.file_uploader("Upload Fake transaction CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Select model
        model_choice = st.selectbox("Select Model", ["Isolation Forest", "XGBoost"])

        # Predict button
        if st.button("Detect Fraud"):
            if model_choice == "Isolation Forest":
                results = predict_isolation_forest(df)
            else:
                results = predict_xgboost(df)

            # Display the entire dataframe with predictions
            st.write("### Prediction Results")
            st.dataframe(results)  # This will show the entire dataframe including the 'prediction' column
#################################################################
# Show selected page
# st.title("Credit Card Fraud Detection")
if st.session_state.page == "overview":
    show_data_overview()
elif st.session_state.page == "analysis":
    show_data_analysis()

elif st.session_state.page == "customer":
    show_customer_analysis()

elif st.session_state.page == "datetime":
    show_data_distribute()

elif st.session_state.page == "model":
    show_data_model()