import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression

st.set_page_config(
    page_title="Datmin - CO2 EMISSIONS",
    page_icon=":car:",
)

# Load the CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function for EDA (Exploratory Data Analysis)
def perform_eda(df):
    st.title('EDA (Exploratory Data Analysis)')
    st.write("""Pada halaman ini menampilkan visualisasi :
    - Heatmap (untuk melihat korelasi antar fitur) 
    - Distribusi (untuk melihat distribusi jumlah data dari masing-masing fitur)
    - Relasi (untuk melihat hubungan antar dua fitur)
    - Komposisi (untuk melihat komposisi fitur dalam bentuk piechart)
    - Perbandingan (untuk melihat perbandingan antara fitur-fitur)
    """)

    # Create a grid layout for heatmap, distribution, composition, and comparison
    col1, col2 = st.columns(2)
    
# Heatmap
    with col1:
        st.subheader('Visualisasi Heatmap CO2 Emissions dengan Fitur Lainnya')
        fig, ax = plt.subplots(figsize=(6, 4))  # Perkecil visualisasi heatmap
        sns.heatmap(df.corr(), annot=True, cmap='Reds', fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
        st.markdown('Berdasarkan heatmap, dapat disimpulkan bahwa fitur yang paling mempengaruhi nilai gas CO2 Emissions adalah fuelconsumtion_comb dan fuelconsumption_hwy')

    # Distribution
    with col2:
        st.subheader('Distribusi Variabel')
        selected_variable = st.selectbox('Pilih Variabel', df.columns)
        if selected_variable:
            fig, ax = plt.subplots(figsize=(6, 4))  # Perkecil visualisasi distribusi variabel
            sns.histplot(df[selected_variable], kde=True, ax=ax)
            ax.set_xlabel(selected_variable)
            ax.set_ylabel('Frekuensi')
            ax.set_title(f'Distribusi {selected_variable}')
            st.pyplot(fig)
            if selected_variable == 'ENGINESIZE':
                st.markdown("Distribusi dari **Engine Size**. Ini adalah distribusi dari ukuran mesin kendaraan. Dan dapat dilihat nilai ukuran mesin 4 paling tinggi distribusi datanya")
            elif selected_variable == 'CYLINDERS':
                st.markdown("Distribusi dari **Cylinders**. Ini adalah distribusi dari jumlah silinder kendaraan. Dan dapat dilihat jumlah silinder 4 dan 6 yang paling tinggi distribusi datanya ")
            elif selected_variable == 'FUELTYPE':
                st.markdown("Distribusi dari **Fuel Type**. Ini adalah distribusi dari jenis bahan bakar kendaraan. Dan bisa dilihat bahwa jenis bahan bakar bensin dan diesel yang paling banyak digunakan oleh kendaraan")          
            elif selected_variable == 'FUELCONSUMPTION_CITY':
                st.markdown("Distribusi dari **Fuel Consumption in City**. Ini adalah distribusi dari konsumsi bahan bakar di kota. Dan dapat dilihat bahwa rata-rata konsumsi bahan bakar dikota paling tinggi yaitu 10 Liter")
            elif selected_variable == 'FUELCONSUMPTION_HWY':
                st.markdown("Distribusi dari **Fuel Consumption on Highway**. Ini adalah distribusi dari konsumsi bahan bakar di jalan raya. Dan dapat dilihat rata-rata konsumsi bahan bakar di jalan raya paling tinggi yaitu 8 liter")
            elif selected_variable == 'FUELCONSUMPTION_COMB':
                st.markdown("Distribusi dari **Fuel Consumption Combined**. Ini adalah distribusi dari konsumsi bahan bakar yang dikombinasikan. Dan dapat dilihat rata-rata konsumsi bahan bakar yang dikombinasikan di kota dan di jalan raya yang paling tinggi yaitu 10 Liter")
            elif selected_variable == 'CO2EMISSIONS':
                st.markdown("Distribusi dari **C02 EMISSIONS**. Ini adalah distribusi dari Gas CO2 Emissions. Dan dapat dilihat bahwa gas CO2 yang dihasilkan oleh kendaaraan yang paling tinggi bernilai 250")
        else:
            st.write("Pilih sebuah variabel untuk melihat distribusi.")
    # Panggil fig, ax = plt.subplots() dan st.pyplot(fig)
    # Scatter Plot with Linear Regression
    st.subheader('Relasi dengan Garis Linear')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Perkecil ukuran scatter plot

    sns.regplot(x='ENGINESIZE', y='CO2EMISSIONS', data=df, color='blue', line_kws={"color": "red"}, ax=axes[0, 0])
    axes[0, 0].set_xlabel('ENGINE SIZE')
    axes[0, 0].set_ylabel('CO2 Emission')
    axes[0, 0].set_title('ENGINESIZE vs CO2 Emission')

    sns.regplot(x='CYLINDERS', y='CO2EMISSIONS', data=df, color='orange', line_kws={"color": "red"}, ax=axes[0, 1])
    axes[0, 1].set_xlabel('CYLINDERS')
    axes[0, 1].set_ylabel('CO2 Emission')
    axes[0, 1].set_title('CYLINDERS vs CO2 Emission')

    sns.regplot(x='FUELCONSUMPTION_CITY', y='CO2EMISSIONS', data=df, color='green', line_kws={"color": "red"}, ax=axes[1, 0])
    axes[1, 0].set_xlabel('FUELCONSUMPTION_CITY')
    axes[1, 0].set_ylabel('CO2 Emission')
    axes[1, 0].set_title('FUELCONSUMPTION_CITY vs CO2 Emission')

    sns.regplot(x='FUELCONSUMPTION_HWY', y='CO2EMISSIONS', data=df, color='yellow', line_kws={"color": "red"}, ax=axes[1, 1])
    axes[1, 1].set_xlabel('FUELCONSUMPTION_HWY')
    axes[1, 1].set_ylabel('CO2 Emission')
    axes[1, 1].set_title('FUELCONSUMPTION_HWY vs CO2 Emission')

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('Berdasarkan dari visualisasi scatter plot di atas didapatkan insight bahwa nilai fitur fuelconsumption_city paling mendekati dengan garis linear yang artinya paling mempengaruhi nilai CO2 Emissions')

    # Composition
    st.subheader('Komposisi dan Perbandingan Variabel')
    # Create a grid layout for composition and comparison
    col3, col4 = st.columns(2)


    # Pie chart for fuel type composition
    with col3:
        st.markdown('**Komposisi Jenis Bahan Bakar**')
        fueltype_composition = df['FUELTYPE'].value_counts()
        fueltype_chart = plt.figure(figsize=(6, 6))  # Perkecil ukuran visualisasi pie chart
        plt.pie(fueltype_composition, labels=fueltype_composition.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Komposisi Jenis Bahan Bakar')
        st.pyplot(fueltype_chart)
        st.write("""Jenis bahan bakar  :
                    - 0 (Bensin Pertamax) 
                    - 1 (Etanol(Campuran Etanol-Bensin))
                    - 2 (Bensin Premium)
                    - 3 (Diesel (Solar))
                """)
        st.markdown('Dapat dilihat dari visualisasi pie chart di atas di dapatkan insight bahwa jenis bahan bakar yang paling banyak digunakan oleh kendaraan mobil adalah bahan bakar diesel (0). Sedangkan bahan bakar Etanol (1) yang paling sedikit digunakan oleh kendaraan mobil')

    # Bar chart for comparison of fuel consumption
    with col4:
        st.markdown('**Perbandingan Konsumsi Bahan Bakar**')
        fuel_consumption_columns = ['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']
        fuel_consumption_comparison = df[fuel_consumption_columns].mean()

        fuel_consumption_chart = plt.figure(figsize=(6, 6))  # Perkecil ukuran visualisasi bar chart
        fuel_consumption_comparison.plot(kind='bar', color=['green', 'blue', 'purple'])
        plt.title('Perbandingan Konsumsi Bahan Bakar')
        plt.xlabel('Tipe Konsumsi Bahan Bakar')
        plt.ylabel('Rata-rata Konsumsi Bahan Bakar')
        plt.xticks(rotation=45)
        st.pyplot(fuel_consumption_chart)
        st.markdown('Dari visualisasi di atas, kita dapat melihat rata-rata konsumsi bahan bakar untuk berbagai jenis penggunaan jalan. Dan dapat disimpulkan bahwa rata-rata konsumsi bahan bakar di kota yang paling tinggi')

        st.markdown('Dari visualisasi di atas, kita dapat melihat rata-rata konsumsi bahan bakar untuk berbagai jenis penggunaan jalan. Dan dapat disimpulkan bahwa rata-rata konsumsi bahan bakar di kota yang paling tinggi')

# Function for prediction
def predict():
    df = pd.read_csv('data_final.csv')
    st.write("""
    # Prediksi Gas CO2 Emissions
    Gunakan model di bawah untuk memasukkan fitur kendaraan dan memprediksi emisi CO2 gas.
    """)
    st.subheader("Dataset CO2 Emissions yang telah dicleaning")
    st.write(df)  # Menampilkan keseluruhan dataset dari file data_final.csv

    x = df.drop('CO2EMISSIONS', axis=1)
    y = df['CO2EMISSIONS']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    loaded_model = joblib.load(open('model_linear_fix.pkl', 'rb'))

    # Input features
    ENGINESIZE = st.number_input('Input nilai ENGINESIZE')
    CYLINDERS = st.number_input('Input nilai CYLINDERS')
    fueltype_options = ['Z', 'E', 'X', 'D']
    FUELTYPE = st.selectbox('Select FUELTYPE', fueltype_options)
    FUELCONSUMPTION_CITY = st.number_input('Input nilai FUELCONSUMPTION_CITY')
    FUELCONSUMPTION_HWY = st.number_input('Input nilai FUELCONSUMPTION_HWY')
    FUELCONSUMPTION_COMB = st.number_input('Input nilai FUELCONSUMPTION_COMB')

    # Define input_data
    input_data = pd.DataFrame({
        'ENGINESIZE': [ENGINESIZE],
        'CYLINDERS': [CYLINDERS],
        'FUELTYPE': [FUELTYPE],
        'FUELCONSUMPTION_CITY': [FUELCONSUMPTION_CITY],
        'FUELCONSUMPTION_HWY': [FUELCONSUMPTION_HWY],
        'FUELCONSUMPTION_COMB': [FUELCONSUMPTION_COMB]
    })

    # Prediction
    if st.button('Prediksi Gas CO2 Emissions'):
        predicted_value = loaded_model.predict(input_data)
        st.write('Prediksi Gas CO2 Emissions Sebesar: ', predicted_value[0]) 


# Main function
def main():
    # Load CSS
    local_css("style.css")

    # Display main menu
    st.sidebar.title('Menu')
    menu_options = ['Home', 'EDA', 'Predict']
    choice = st.sidebar.selectbox('Pilih Menu', menu_options)

    if choice == 'Home':
        st.write("""
        # Welcome to Dashboard Prediksi Gas CO2 Emissions :car:
        """)
        st.image('emisimobil.jpg', use_column_width=True)
        st.write("""
        Dashboard ini memungkinkan Anda untuk melihat hasil analisis dari data Gas CO2 Emissions kendaraan Mobil.
        Dengan dibuatnya dashboard ini, Anda dapat:
        - Melihat visualisasi yang insighful dari data Gas CO2 Emissions kendaraan Mobil.
        - Melihat hasil prediksi Gas CO2 Emissions kendaraan mobil dengan model regresi yang telah dibuat.

        ## Daftar Menu
        1. :house: Home
        2. :bar_chart: Visualisasi EDA
        3. :chart_with_upwards_trend: Predict
        
        ## Penjelasan Singkat Gas CO2
        Gas CO2 (karbon dioksida) adalah gas yang umumnya dihasilkan oleh aktivitas manusia seperti pembakaran bahan bakar fosil, industri, dan pertanian. Gas ini merupakan penyumbang utama terhadap pemanasan global dan perubahan iklim di seluruh dunia. Dalam konteks emisi gas CO2 dari kendaraan bermotor, hal ini menjadi perhatian serius karena mobil merupakan sumber emisi CO2 yang signifikan. Dalam upaya mengurangi dampak negatifnya terhadap lingkungan, perlu dilakukan pemahaman dan analisis yang mendalam terhadap faktor-faktor yang mempengaruhi emisi CO2 dari kendaraan, seperti ukuran mesin, jumlah silinder, jenis bahan bakar, dan efisiensi bahan bakar.
        """, unsafe_allow_html=True)

    elif choice == 'EDA':
        # Load the dataset
        df = pd.read_csv('Data Cleaned.csv')
        perform_eda(df)
    elif choice == 'Predict':
        predict()

if __name__ == "__main__":
    main()
