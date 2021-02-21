import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image

image = Image.open('logo.png')
st.image(image,use_column_width=True)
st.write("DNA true")

st.header('Enter DNA')

sequence_input=">DNA query\nGAACACACADATCATCTACTACGATCGATCAGCGATCAGTCAGCTACGACTAGCAGG"

sequence = st.text_area("Sequence input", sequence_input,height=250)	
sequence = suquence.splitlines()
sequence = sequence[1:]
sequence = ''.join(sequence)
st.write("")
st.header('input (DNA Query)')
sequence
st.header('output (DNA Nucleotide Count)')
st.subheader('1.Print dictionary')
def DNA_nucleotide_count(seq):
	d=dict([
		('A',seq.count('A')),
		('A',seq.count('T')),
		('A',seq.count('G')),
		('A',seq.count('C')),
		])
return d
X=DNA_nucleotide_count(sequence)

X_label=list(X)
X_values=list.(X.values())

X

st.subheader('2.print text')
st.write('There are   '+str(X['A'])+'adenine (A)')
st.write('There are   '+str(X['T'])+'thymine (t)')
st.write('There are   '+str(X['G'])+'adenine (guanine)')
st.write('There are   '+str(X['C'])+'thymine (cytosine)')

st.subheader('3.display dataframe')
df=pd.DataFrame.from_dict(X, orient='index')
df=df.rename({0:'count'},axis='columns')
df.reset_index(inplace=True)
df.df.rename(columns={'index':'nucleotide'})
st.write(df)

st.subheader('4. display bar chart')
p = alt.Chrt(df).mark_bar().encode(
	x='nucleotide',
	y='count')
p=p.properties(width=alt.step(80))
st.write(p)
