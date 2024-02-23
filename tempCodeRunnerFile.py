    pdf = (arrayFlat)/array.shape[0] *array.shape[1]
    cdf = np.zeros(len(arrayFlat))
    # st.write(pdf[1])
    for i in range(1, len(arrayFlat)):
        cdf[i] = pdf[i]+pdf[i-1]
    if(cdf[-1]/100 == 1):
        print("success")
    else:
        print(cdf[-1])
    result = cdf[arrayFlat]