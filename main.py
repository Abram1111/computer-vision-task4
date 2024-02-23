import streamlit as st
from PIL import Image
from pathlib import Path
import streamlit_nested_layout
import cv2
import matplotlib.pyplot as plt
import Filters
import numpy as np
import Histograms as Hs
import Frequency as freq
import pandas as pd
import plotly_express as px
import plotly.figure_factory as ff
from active_contour import *
from FeatureMatching import *
from libs import *
import Sift
import thresholding
from segmentation import *
from region_growing import *

images_folder = 'images'
freq_pic1 = ''
freq_pic2 = ''


st.set_page_config(layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8,tab9 = st.tabs(
    ["Filters", "Histograms", "Frequency", "active contour", "Feature matching", "Harris operator", "SIFT","Threshold","Segmentation"])

with tab1:
    st.header("Filters")
    before_col, after_col = tab1.columns([1, 2])
    before_col2, after_col2 = tab1.columns(2)
    with before_col:
        file = st.file_uploader("Choose Image")
    with after_col:
        type, list, slider = st.columns(3)
        with type:
            opration_type = st.radio(
                "Choose opration", ["Add Noise", "Filter", "Edge Detection"], label_visibility="visible")
        with list:
            ############################## Noise Section ################################
            if opration_type == "Add Noise":
                noise_type = st.radio(
                    "Choose Noise", ["Salt&Papper", "Gaussian", "Uniform"])
                if noise_type == "Salt&Papper":
                    with slider:
                        noise_ratio = st.slider("Noise Ratio", min_value=0, max_value=100,
                                                value=5, step=1,  label_visibility="visible")
                if noise_type == "Gaussian":
                    with slider:
                        gaussuan_sigma = st.slider("Sigma", min_value=0.0, max_value=1.0,
                                                   value=.3, step=.1,  label_visibility="visible")
                if noise_type == "Uniform":
                    with slider:
                        noise_ratio = st.slider("Noise Ratio", min_value=0.0, max_value=1.0,
                                                value=.2, step=.1,  label_visibility="visible")
            ############################## Filter Section ################################
            elif opration_type == "Filter":
                filter_type = st.radio(
                    "Choose filter", ["Avrage Filter", "Gaussian Filter", "Median Filter"])
                with slider:
                    kernal_dim = st.slider("kernal dimention", min_value=2, max_value=10,
                                           value=3, step=1,  label_visibility="visible")
                if filter_type == "Gaussian Filter":
                    with slider:
                        gaussuan_sigma = st.slider("Sigma", min_value=0.0, max_value=1.0,
                                                   value=.3, step=.1,  label_visibility="visible")

            ############################## Edge Section ################################
            elif opration_type == "Edge Detection":
                edge_type = st.radio(
                    "Choose Operator", ["Prewitt Operator", "Sobel Operator", "Roberts Operator", "Canny Operator"])
                if edge_type == "Sobel Operator":
                    with slider:
                        x_dir = st.checkbox("Soble x", False)
                        y_dir = st.checkbox("Soble y", False)

    if file is not None:
        name = 'images\\'+file.name
        img_gray = cv2.imread(name, 0)
        ############################## Noise Section ################################
        if opration_type == "Add Noise":
            img_gray = img_gray/255
            if noise_type == "Salt&Papper":
                final_img = Filters.add_sp_noise(img_gray, (noise_ratio/100))
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
            elif noise_type == "Gaussian":
                final_img = Filters.add_gaussian_noise(
                    img_gray, gaussuan_sigma)
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
            elif noise_type == "Uniform":
                final_img = Filters.add_uniform_noise(
                    img_gray, noise_ratio)
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
        ############################## Filter Section ###############################
        elif opration_type == "Filter":
            if filter_type == "Avrage Filter":
                final_img = Filters.avrage_filter2(img_gray, kernal_dim)
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
            elif filter_type == "Gaussian Filter":
                final_img = Filters.gaussian_filter(
                    img_gray, kernal_dim, gaussuan_sigma)
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
            elif filter_type == "Median Filter":
                final_img = Filters.median_filter(img_gray, kernal_dim)
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
        ############################## Edge Section ##################################
        elif opration_type == "Edge Detection":
            if edge_type == "Roberts Operator":
                final_img = Filters.roberts_edge(img_gray)
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
            elif edge_type == "Prewitt Operator":
                final_img = Filters.Prewitt_edge(img_gray)
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
            elif edge_type == "Sobel Operator":
                final_img = Filters.Sobel_edge(img_gray, x_dir, y_dir)
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
            elif edge_type == "Canny Operator":
                final_img = Filters.canny_edge(img_gray)
            plt.imsave('images\\final_img.png', final_img, cmap='gray')
        with before_col2:
            st.image(file, caption=None, width=500,
                     channels="RGB", output_format="auto")

        with after_col2:
            final_img = cv2.imread('images\\final_img.png', 0)
            st.image(final_img, caption=None, width=500,
                     channels="GRAY", output_format="auto")
            save = st.button("save")
            if save:
                name = 'images\\save_img' + \
                    format(np.random.randint(1, 100))+'.png'
                plt.imsave(name, final_img, cmap='gray')
with tab2:
    st.header("Histograms")
    # st.sidebar.title("Select Image")
    before, after = tab2.columns(2)
    with before:
        file = st.file_uploader('choose img')
    with after:
        opration_type = st.radio(
            "Choose opration", ["Histogram Equalization", "Normalize", "Thresholding", "Convert To Gray"], horizontal=True)
    before1, after1 = tab2.columns(2)
    # up, down = tab2.rows(2)
    # st.sidebar.button("Convert To Grey")
    if file is not None:
        data1 = cv2.imread('images\\'+file.name)
        # st.write("ImageA colored", data1.shape)
        data = cv2.imread('images\\'+file.name, 0)
        # print("ImageA ", data.shape)
        hist = cv2.calcHist(data, [0], None, [256], [0, 256])
    if(opration_type == "Histogram Equalization" and (file is not None)):
        equalized = Hs.histEqualization(data, max(data.ravel()))
        with before1:
            st.image('images\\'+file.name, caption='Before')
            fig = plt.figure()
            plt.hist(data.ravel(), 256, [0, 256])
            # plt.hist(data.ravel())
            st.plotly_chart(fig)
            unique, counts = np.unique(data.ravel(), return_counts=True)
            # st.title("Before")
            fig2 = px.line(
                x=unique,
                y=counts,
                color_discrete_sequence=['red'], labels={"x": "Gray Level", "y": "No of Pixels"}
            )
            fig2['data'][0]['showlegend'] = True
            fig2['data'][0]['name'] = 'before'
            unique1, counts1 = np.unique(equalized, return_counts=True)
            fig2.add_scatter(name="after", x=unique1,
                             y=counts1, line_color="green")
            st.plotly_chart(fig2)
        with after1:
            fig3 = plt.figure()
            # st.write(data)

            plt.hist(equalized, 256, [0, 256])
            equalizedImage = np.reshape(equalized, data.shape)
            cv2.imwrite("images\\HistEqualized.png", equalizedImage)
            st.image("images\\HistEqualized.png", caption='After')
            st.plotly_chart(fig3)
            fig5 = plt.figure()
            a = Hs.drawCumulativeEq(data, equalizedImage)
            st.plotly_chart(a)
    if(opration_type == "Normalize" and (file is not None)):
        with before1:
            norm = Hs.Normalize(data)
            cv2.imwrite("images\\Normalized.png", norm)
            # st.write("Normalized Image")
            st.image("images\\Normalized.png", caption='Normalized Image')
        with after1:
            # st.write("Original")
            st.image('images\\'+file.name, caption='Original')

    if(opration_type == "Thresholding" and (file is not None)):
        with before1:
            st.image('images\\'+file.name, caption='Original Image')
        with after1:
            glob, loc = st.columns(2)
            with glob:
                array = Hs.Thresholding(data, 255, 0)
                cv2.imwrite("images\\Thresholded.png", array)
                st.image("images\\Thresholded.png",
                         caption='Global Thresholding')

            with loc:
                array1 = Hs.localThresholding(data, [2, 2])
                array1 = np.reshape(array1, data.shape)
                cv2.imwrite("images\\localThresholded.png", array1)
                st.image("images\\localThresholded.png", 'Local Thresholding')

    if(opration_type == "Convert To Gray" and (file is not None) and data1[0, 0, 2] != data1[0, 0, 1] != data1[0, 0, 0]):
        if(len(data1.shape) == 3):
            with before1:
                st.image('images\\'+file.name,
                         caption="The RGB Image", width=475)
                unique2, counts2 = np.unique(data1[..., 0], return_counts=True)
                print(data1[..., 0].shape)
                # fig0 = plt.figure()
                # plt.plot(unique2, counts2)
                # plt.hist(data1[...,0].ravel(), 256, [0, 256], color='r', alpha = 0.9)
                # plt.hist(data1[..., 0].ravel(), 256, [0, 256], color='r')
                # st.plotly_chart(fig0, use_container_width=True)
            # with TestEdit:
                # fig0 = plt.figure()
                # plt.hist(data1[..., 1].ravel(), 256, [0, 256], color='g')
                # st.plotly_chart(fig0, use_container_width=True)
                unique3, counts3 = np.unique(data1[..., 1], return_counts=True)
                a = Hs.drawCumulative1(data1)
                st.plotly_chart(a, use_container_width=True)
                # a = Hs.drawCumulative(data1[...,1], 'green')
                # st.plotly_chart(a)

            # with after:
            with after1:
                greyImage = Hs.ToGrey(data1)
                print(data.shape)
                cv2.imwrite("images\\GreyScale.png", greyImage)
                st.image("images\\GreyScale.png",
                         caption="The Gray Scale Image", width=475)
                # fig0 = plt.figure()
                # plt.hist(data1[..., 2].ravel(), 256, [0, 256], color='b')

                fig = px.line(
                    x=unique2,
                    y=counts2,   labels={
                        "x": "Gray Level", "y": "No of Pixels"},
                    color_discrete_sequence=['red']
                )
                fig['data'][0]['showlegend'] = True
                fig['data'][0]['name'] = 'Distribution curve of Red'
                fig.add_scatter(name="Distribution curve of Green", x=unique3,
                                y=counts3, line_color="green")
                unique4, counts4 = np.unique(data1[..., 2], return_counts=True)
                fig.add_scatter(name="Distribution curve of Blue", x=unique4,
                                y=counts4, line_color="blue")
                st.plotly_chart(fig, use_container_width=True)
                # st.write(data1[0, 0, 0])
                # st.write(data1[0, 0, 1])
                # st.write(data1[0, 0, 2])

                # hist_data = [data1[..., 0].ravel(
                # ), data1[..., 1].ravel(), data1[..., 2].ravel()]
                # group_labels = ['Red Histogram',
                #                 'Green Histogram', 'Blue Histogram']

                # fig5 = ff.create_distplot(
                #     hist_data, group_labels, bin_size=range(len(data1[..., 2].ravel())))
                # st.plotly_chart(fig5, use_container_width=True)
            df = pd.DataFrame(dict(series=np.concatenate((len(data1[..., 0].ravel())*["Red Histogram"],
                                                          len(data1[..., 1].ravel(
                                                          ))*["Green Histogram"],
                                                          len(data1[..., 2].ravel())*["Blue Histogram"])),
                                   data=np.concatenate((data1[..., 0].ravel(),
                                                        data1[..., 1].ravel(
                                   ),
                                       data1[..., 2].ravel()))))
            fig1 = px.histogram(data_frame=df, x="data", category_orders=dict(data=np.arange(0, 256, 1)), color_discrete_map={
                                "Red Histogram": "red", "Green Histogram": "green", "Blue Histogram": "blue"}, color="series")
            # st.plotly_chart(fig0, use_container_width=True)
            st.plotly_chart(fig1, use_container_width=True)

    else:
        st.error("This image is already a gray scale image")
with tab3:
    st.header("Frequency")
    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        uploaded_image1 = st.file_uploader("Low_pass Image",  key=1)
    with col2:
        uploaded_image2 = st.file_uploader("High_pass Image", key=2)

    if uploaded_image1 is not None:

        # save_path = Path(images_folder, uploaded_image1.name)
        # freq_pic1 = 'images\\'+uploaded_image1.name
        # with open(save_path, mode='wb') as w:
        #     w.write(uploaded_image1.getvalue())
        file1 = 'images\\'+uploaded_image1.name

    if uploaded_image2 is not None:
        # save_path = Path(images_folder, uploaded_image2.name)
        # freq_pic2 = 'images\\'+uploaded_image2.name
        # with open(save_path, mode='wb') as w:
        #     w.write(uploaded_image2.getvalue())
        file2 = 'images\\'+uploaded_image2.name

    if uploaded_image1 is not None:
        if uploaded_image2 is not None:
            # image1 = cv2.imread(file1,0)
            image1 = cv2.imread(file1)
            image1 = Hs.ToGrey(image1)
            image2 = cv2.imread(file2)
            image2 = Hs.ToGrey(image2)
            # result = freq.hypird_image(image1, image2)

            # lowpass, highpass = st.columns(2)
            # with lowpass:
            #     st.image('debug/original_low.jpg')
            #     st.image('debug/filterd_low.jpg')
            # with highpass:
            #     st.image('debug/original_high.jpg')
            #     st.image('debug/filterd_high.jpg')
            # col, result, col = st.columns([3, 3, 3])
            # with result:
            #     st.image('debug/hypird.jpg')
            with col3:

                sigma = st.slider("standard deviation", min_value=0.1, max_value=20.0,
                                  value=10.0, step=0.1,  label_visibility="visible")
                freq.fft_hyprid_image(image1, image2, sigma)

            # lowpass, highpass = st.columns(2)
            # with lowpass:
            #     st.image('debug/lgauss.jpg')
            #     # st.image('debug/filterd_low.jpg')
            # with highpass:
            #     st.image('debug/hgauss.jpg')
            #     # st.image('debug/filterd_high.jpg')

            original, filter, result = st.columns([2, 2, 4])
            with original:
                st.image('debug/original_low.jpg')
                st.image('debug/original_high.jpg')

            with filter:
                st.image('debug/lgauss.jpg')
                st.image('debug/hgauss.jpg')

            with result:
                empty3, hybrid_col, empty4 = st.columns([1, 7, 1])
                with hybrid_col:
                    empty1, text, empty2 = st.columns([1.5, 3, 1])
                    with text:
                        st.subheader('Hybrid Image')
                    st.image('debug/gauss.jpg')
with tab4:

    st.header("Active contour")
    file = st.file_uploader("Active Image", key=10)
    alpha_slider, beta_slider, gamma_slider = st.columns(3)
    before_col, after_col = tab4.columns([1, 1])

    if file is not None:

        with alpha_slider:
            alpha = st.slider("Alpha",  min_value=0.0, max_value=2.0,
                              value=1.0, step=0.1,  label_visibility="visible", key='a')
        with beta_slider:
            beta = st.slider("Beta", min_value=0.0, max_value=2.0,
                             value=1.0, step=0.1,  label_visibility="visible", key='b')
        with gamma_slider:
            gamma = st.slider("Gamma", min_value=0.0, max_value=2.0,
                              value=1.0, step=0.1,  label_visibility="visible", key='g')
        name = 'activecontour_images/'+file.name
        img_gray = cv2.imread(name, 0)

        contour_points = [(88, 156), (70, 106), (144, 151), (160, 156), (170, 110),
                          (190, 156), (205, 155), (241, 155), (189, 224), (135, 235)]

        img = np.array(img_gray)  # reading image
        lap = np.array(img_laplacian(img))  # apply canny
        lap = add_border(lap)  # padding image

        # passing the contour points to contour class
        contour = Contour(contour_points)

        # Create series of images fitting contour
        allimgs = []
        iteration_num = 100
        for i in range(iteration_num):

            lapcpy = np.copy(lap)
            contour.calc_energies(lapcpy, alpha, beta, gamma)
            contour.update_points()
            contour.draw_contour(lapcpy)  # drawing contour points
            allimgs.append(lapcpy)  # end of loop result

        with before_col:
            st.image(name)
            arr = np.array(contour_points)
            area = cv2.contourArea(arr)
            primeter = cv2.arcLength(arr, closed=True)
            st.write(f"intial primeter:{primeter}")
            st.write(f"intial area:{area}")

        with after_col:
            cv2.imwrite('activecontour_images/result_active.png', allimgs[-1])
            st.image('activecontour_images/result_active.png')
            final = contour.get_contour_points()
            arr = np.array(final)
            area = cv2.contourArea(arr)
            primeter = cv2.arcLength(arr, closed=True)
            st.write(f"final primeter:{primeter}")
            st.write(f"final area:{area}")

        st.write("Chain Code represntation")
        code = contour.get_chain_code()
        st.write(code)
with tab5:
    st.header("Feature Matching")
    col1, col2, col3 = st.columns([4, 4, 2.7])

    with col1:
        uploaded_image1 = st.file_uploader(" Image",  key="f1")
    with col2:
        uploaded_image2 = st.file_uploader(" Target", key="f2")
    with col3:

        radio_output = st.radio("Select method",
                                options=("Normalized cross corelation", "SSD"))
        threshold=st.slider(label="Threshold",min_value= 0,max_value=200, value=75)
        resize=st.checkbox(label='resize image',help='resize image is calculated faster but maybe inaccurate')
        Sift_builttIn=st.checkbox(label='Sift Built In',help='Built in is faster')

    if uploaded_image1 is not None:
        # st.write(uploaded_file)

        save_path = Path('Template', uploaded_image1.name)
        freq_pic1 = 'Template\\'+uploaded_image1.name
        with open(save_path, mode='wb') as w:
            w.write(uploaded_image1.getvalue())
        file1 = 'Template\\'+uploaded_image1.name
        with col1:
                st.image(file1)
    if uploaded_image2 is not None:
        save_path = Path('Template', uploaded_image2.name)
        freq_pic2 = 'Template\\'+uploaded_image2.name
        with open(save_path, mode='wb') as w:
            w.write(uploaded_image2.getvalue())
        file2 = 'Template\\'+uploaded_image2.name
        with col2:
                st.image(file2)

    if uploaded_image1 is not None:
        if uploaded_image2 is not None:

                image1 = cv2.imread(file1)
                target = cv2.imread(file2)
                time = call_matching(image1, target,radio_output,threshold,resize,Sift_builttIn)

                with col3:
                    st.image('template/result.png')
                st.header(f"Execution Time = {time}")
          
with tab6:
    st.write("Harris operator")
    uploaded_file = st.file_uploader("image")
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        # Load the image
        with col1:
            st.image(uploaded_file, caption="Original Image",
                     use_column_width=True)
        # read the uploaded file as bytes
        image_bytes = uploaded_file.read()

        # use cv2 to decode the image bytes
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

        # Detect corners using Harris corner detector
        corners = harris_corner_detection(image)

        # Draw circles at the detected corners
        for x, y in corners:
            cv2.circle(image, (x, y), 20, (0, 0, 255), -1)

        # Display the image with the detected corners
        with col2:
            st.image(image, channels="BGR",
                     caption="Harris Image", use_column_width=True)
with tab7:
    SIFT_FILE = st.file_uploader("Choose Image", key=100)
    SIFT1, SIFT2 = tab7.columns(2)

    if SIFT_FILE is not None:
        name = 'images\\' + SIFT_FILE.name
        img_gray = cv2.imread(name, 0)
        with SIFT1:
            st.image(SIFT_FILE, caption=None, width=500,
                     channels="RGB", output_format="auto")
        final_img = Sift.apply_sift(img_gray)
        plt.imsave('images\\final_img.png', final_img)
        with SIFT2:
            final_img,time= cv2.imread('images\\final_img.png')
            st.image(final_img, caption=None, width=500,
                     channels="RGB", output_format="auto")
            st.header(f"Execution Time = {time}")
with tab8:
    st.header("Threshold")
    before_col, after_col = tab8.columns(2)
    before_col1, after_col2 = tab8.columns(2)
    with before_col:
         file = st.file_uploader("Choose Image",key=400)
    with after_col:
          opration_type = st.radio(
                "Choose opration", ["Optimal Thresholding", "Otsu Thresholding", "Spectral Thresholding","Optimal Thresholding using local thresholding concept", "Otsu Thresholding using local thresholding concept","Spectral Thresholding using local thresholding concept","Local Thresholding"], label_visibility="visible")
    
    if file is not None:
        name = 'images\\' + file.name
        img_gray = cv2.imread(name, 0)
        with before_col1:
            st.image(file, width=500,
                     channels="RGB", output_format="auto", caption="Before Thresholding")
        if opration_type == "Optimal Thresholding":
             final_img,threshold = thresholding.iterative_thresholding(img_gray)
             plt.imsave('images\\final_img.png', final_img, cmap='gray')
        elif opration_type == "Otsu Thresholding":
             final_img,threshold = thresholding.otsu_threshold(img_gray)
             plt.imsave('images\\final_img.png', final_img, cmap='gray')
        elif opration_type == "Spectral Thresholding":
             final_img = thresholding.spectral_thresholding(img_gray)
             plt.imsave('images\\final_img.png', final_img, cmap='gray')
        elif opration_type == "Otsu Thresholding using local thresholding concept":
                final_img = thresholding.otsu_threshold_local(img_gray, [500,500])
                plt.imsave('images\\final_img.png', final_img, cmap='gray')
        elif opration_type == "Optimal Thresholding using local thresholding concept":
             final_img = thresholding.iterative_thresholding_local(img_gray, [500,500])
             plt.imsave('images\\final_img.png', final_img, cmap='gray')
        elif opration_type == "Spectral Thresholding using local thresholding concept":
             final_img = thresholding.spectral_thresholding_local(img_gray, [500,500])
             plt.imsave('images\\final_img.png', final_img, cmap='gray')
        elif opration_type == "Local Thresholding":
             final_img = thresholding.localThresholding(img_gray, [3,3])
             plt.imsave('images\\final_img.png', final_img, cmap='gray')
        with after_col2:
            final_img = cv2.imread('images\\final_img.png', 0)
            st.image(final_img, width=500,
                     channels="GRAY", output_format="auto", caption="After Thresholding")
with tab9:
    st.header("Unsupervised Segmentation")
    col1, col2,col3 = st.columns([4, 4, 4])
    input,output=st.columns([2,2])

    with col1:
        uploaded_image1 = st.file_uploader(" Image",  key="seg")
    with col2:

        radio_output = st.radio("Select method",
                                options=("Kmeans", "Region Growing", "Agglomerative", "Mean Shift"))

    if uploaded_image1 is not None:
        save_path = Path('Images', uploaded_image1.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_image1.getvalue())
        org_image = 'Images\\'+uploaded_image1.name
        with input:
                st.image(org_image)

    if uploaded_image1 is not None:
                org_image = cv2.imread(org_image)
                segmented_img=np.zeros((3,3))
                with col3:
                    if radio_output=="Kmeans":
                        k=st.slider(label="K Clusters",min_value= 1,max_value=30, value=5)
                        threshold=st.slider(label="Threshold",min_value= 0.0,max_value=2.0, value=0.85,key="kmeans")
                        segmented_img=kmeans(org_image,k)
                    elif radio_output=="Region Growing":

                        threshold=st.slider(label="Threshold",min_value= 0.0,max_value=100.0, value=10.0,key="REGgrow")
                        neighboursNum = st.radio("Select Neighbours Num",options=("4", "8"))
                        
                        segmented_img=run_region_growing('Images\\'+uploaded_image1.name ,float(threshold) ,int(neighboursNum) )

                    elif radio_output=="Agglomerative":
                        # Load the image and resize it
                        org_image = transform.resize(org_image, (org_image.shape[0]/4 , org_image.shape[1]/4 ))

                        # Extract feature vectors from the image
                        features = org_image.reshape(-1, 3)  # assuming the image is in RGB format


                        # Apply agglomerative clustering with approximate nearest neighbors
                        agg = AgglomerativeClustering()
                        agg.fit(features)

                        # Assign each pixel to its corresponding cluster
                        labels = agg.labels_.reshape(org_image.shape[:2])

                        # Visualize the segmentation
                        segmented_img = mark_boundaries(org_image, labels)
                        segmented_img = transform.resize(segmented_img, (segmented_img.shape[0]*4 , segmented_img.shape[1]*4 ))


                    elif radio_output=="Mean Shift":
                        window=st.slider(label="Window Size",min_value= 1,max_value=100, value=70)
                        threshold=st.slider(label="Threshold",min_value= 0.0,max_value=2.0, value=1.0,key="meanshift")
                        segmented_img=mean_shift(org_image,window,threshold)

                with output:
                    st.write(" ")
                    st.image(segmented_img)
