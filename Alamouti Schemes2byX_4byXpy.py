# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 07:27:33 2021

@author: aguboshimec
"""
import numpy as np
from numpy import sqrt
import random
import matplotlib.pyplot as plt

NoOfSamples = 50 #no. of observations or samples
EbNodB_range = range(0,10)
itr = len(EbNodB_range) #iterate over this range of EbNo
ber = [None]*itr
ber_2by1 = [None]*itr
ber_4by2 = [None]*itr
ber_uncoded = [None]*itr
ber4by4_Uncoded = [None]*itr
ebNo = []
err2by2 = []
ep2by2 = []
errUncoded = []
epUncoded = []
err2by1 = []
ep2by1 = []
h = [] #stores a single channel coefficient


def Alamouti_2byX ():
    for k in range (0, itr):  
        EbNodB = EbNodB_range[k]   
        EbNo=10.0**(EbNodB/10.0)
        noise_std = 1/sqrt(2*EbNo)
        noise_mean = 0    
        no_errors = 0 #initialize number of errors.
        no_errors_uncoded = 0
        no_errors2by1 = 0
        for m in range (0, NoOfSamples):
            n = []
            tx_symbol1 = (2*random.randint(0,1)-1) #generate first BPSK symbols
            tx_symbol2 = (2*random.randint(0,1)-1) #generate second BPSK symbols
            symbol = np.array([tx_symbol1, tx_symbol2]) #concatenates both. Forms a vector
                                  
            for j in range(4): #since, I am considering a 2by2 Alamouti Scheme, ie. 4 distinct Channel Coefficients
                hxx =  np.around(np.random.randn() - np.random.randn()*1j, 5) #hxx means h11, h12, etc
                h.append(hxx)
                #noise: the channel impairment had to be within some predefined mean and std-dev
                nxx = np.random.normal(loc=noise_mean, scale=noise_std, size=None) - np.random.normal(loc=noise_mean, scale=noise_std, size=None)*1j ##specifically, noise with a pre-defined mean & standard deviation    
                n.append(nxx)
                
            H = np.array([[h[0], h[1]], [h[2], h[3]],  [-h[1].conj(), h[0].conj()], [-h[3].conj(), h[2].conj()]]) #i kinda manually concatenated the coefficient to assumed the form for an Alamouti Coding
            H_uncoded = np.array([[h[0], h[1]], [h[2], h[3]]]) #i kinda manually concatenated the coefficient to assumed the form for an Alamouti Coding

            H2by1 = np.array([[h[0], h[1]], [h[1].conj(), -h[0].conj()]]) #forms the coded channel matrix from the previosuly obtained matrix
            
            n = np.array(n)
            Noise = [n[0], n[1], n[2].conj(), n[3].conj()] #noise vector for 2by2
            Noise_uncoded = [n[0], n[1]] #noise vector for 2by2 Uncoded
            Noise2by1 = [n[0], n[1].conj()] #noise vector for 2by1
                 
            rx_symbol =  np.add((np.dot(H, (np.dot((1/sqrt(2)),symbol)))), Noise) #recieved_symbol - 2by2 Alamouti, y == rx_symbol
             
            rx_symbol_uncoded =  np.add((np.dot(H_uncoded, (np.dot((1/sqrt(2)),symbol)))), Noise_uncoded) #recieved_symbol - 2by2 Alamouti, y == rx_symbol
      
            rx_symbol2by1 =  np.add((np.dot(H2by1, (np.dot((1/sqrt(2)),symbol)))), Noise2by1) #recieved_symbol for 2by1 Alamouti
            
            
            #zero_forcing decoder generates the estimate of the transmitted matrix as: x: = pinv(H)*y
            Symbol_est = np.dot(np.linalg.pinv(H), rx_symbol)
            Symbol_est_uncoded = np.dot(np.linalg.pinv(H_uncoded), rx_symbol_uncoded)
            Symbol_est2by1 = np.dot(np.linalg.pinv(H2by1), rx_symbol2by1)
            
            #detection for 2by2 alamouti
            det_symbol1 = 2*(Symbol_est[0].real >= 0) - 1
            det_symbol2 = 2*(Symbol_est[1].real >= 0) - 1
            
            #detection for 2by2 alamouti
            det_symbol1_uncoded = 2*(Symbol_est_uncoded[0].real >= 0) - 1
            det_symbol2_uncoded = 2*(Symbol_est_uncoded[1].real >= 0) - 1
            
            #detection for 2by1 alamouti
            det_symbol1_2by1 = 2*(Symbol_est2by1[0].real >= 0) - 1
            det_symbol2_2by1 = 2*(Symbol_est2by1[1].real >= 0) - 1
            
            no_errors += 1*(tx_symbol1 != det_symbol1)+1*(tx_symbol2 != det_symbol2)  #2by2
            no_errors_uncoded += 1*(tx_symbol1 != det_symbol1_uncoded)+1*(tx_symbol2 != det_symbol2_uncoded)  #2by2
            no_errors2by1 += 1*(tx_symbol1 != det_symbol1_2by1) + 1*(tx_symbol2 != det_symbol2_2by1) #2by1
              
        ber[k] = 1.0*no_errors/(2*NoOfSamples)
        ber_uncoded[k] = 1.0*no_errors_uncoded/(2*NoOfSamples)
        ber_2by1[k] = 1.0*no_errors2by1/(2*NoOfSamples)
        
        err2by2.append(no_errors)
        ep2by2.append(ber[k])
        errUncoded.append(no_errors_uncoded)
        epUncoded.append(ber_uncoded[k])
        err2by1.append(no_errors2by1)
        ep2by1.append(ber_2by1[k])
        ebNo.append(EbNo)
        
        print ("EbNodB:", EbNodB)
        print ("Numbder of errors 2by2:", no_errors)
        print ("Numbder of errors Uncoded:", no_errors_uncoded)
        print ("Numbder of errors 2by1:", no_errors2by1)
        print ("Error probability 2by2:", ber[k])
        print ("Error probability Uncoded:", ber_uncoded[k])
        print ("Error probability 2by1:", ber_2by1[k])
        
plt.plot(ep2by2, ebNo)
plt.plot(ep2by1, ebNo)
plt.plot(epUncoded, ebNo)
plt.xscale('linear')
plt.yscale('log')
plt.ylabel('EbNo(dB)')
plt.xlabel('BER')
plt.legend(['2by2 Alamouti','2by1 Alamouti', 'Uncoded-2by2'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.title('BPSK with Alamouti Scheme')
plt.show()   
  
Alamouti_2byX()  


     
def Alamouti_4byX():
    
    for k in range (0, itr):  
        EbNodB = EbNodB_range[k]   
        EbNo=10.0**(EbNodB/10.0)
        noise_std = 1/sqrt(2*EbNo)
        noise_mean = 0    
        no_errors = 0 #initial number of errors.
        no_errors4by2 = 0
        no_errors_Uncoded = 0
        
        for m in range (0, NoOfSamples):
            n = []
            tx_symbol1 = (2*random.randint(0,1)-1) #generate first BPSK symbols
            tx_symbol2 = (2*random.randint(0,1)-1) #generate second BPSK symbols
            tx_symbol3 = (2*random.randint(0,1)-1) #generate third BPSK symbols, 
            tx_symbol4 = (2*random.randint(0,1)-1) #generate fourth BPSK symbols
            
            symbol = np.array([tx_symbol1, tx_symbol2, tx_symbol3, tx_symbol4]) #concatenates both. Forms a vector
            l = len(symbol)                      
            for j in range(16): #since, I am considering a 4by4 Alamouti Scheme, ie. 16 distinct Channel Coefficients
                hxx =  np.around(np.random.randn() - np.random.randn()*1j, 5) #hxx means h11, h12, etc
                h.append(hxx)
                #noise:
                nxx = np.random.normal(loc=noise_mean, scale=noise_std, size=None) - np.random.normal(loc=noise_mean, scale=noise_std, size=None)*1j              
                n.append(nxx)
           
            H = np.array([[h[0], h[1], h[2], h[3]], [h[4], h[5], h[6], h[7]], [h[8], h[9], h[10], h[11]], [h[12], h[13], h[14], h[15]], 
                         [-h[1].conj(), h[0].conj(), -h[3].conj(), h[2].conj()], [-h[5].conj(), h[4].conj(), -h[7].conj(), h[6].conj()], [-h[9].conj(), h[8].conj(), -h[11].conj(), h[10].conj()], [-h[13].conj(), h[12].conj(), -h[15].conj(), h[14].conj()], 
                         [-h[2].conj(), -h[3].conj(), h[0].conj(), h[1].conj()], [-h[6].conj(), -h[7].conj(), h[4].conj(), h[5].conj()], [-h[10].conj(), -h[11].conj(), h[8].conj(), h[9].conj()], [-h[14].conj(), -h[15].conj(), h[12].conj(), h[13].conj()], 
                            [h[3], -h[2], -h[1], h[0]], [h[7], -h[6], -h[5], h[4]], [h[11], -h[10], -h[9], h[8]], [h[15], -h[14], -h[13], h[12]] ]) #i kinda manually concatenated the coefficient to assumed the form for an Alamouti Coding. Essentially, H is made to be orthogonal
            H4by4_Uncoded = np.array([[h[0], h[1], h[2], h[3]], [h[4], h[5], h[6], h[7]], [h[8], h[9], h[10], h[11]], [h[12], h[13], h[14], h[15]]])
            
            H4by2 = np.array([[h[0], h[4], h[8], h[12]], [h[1], h[5], h[9], h[13]], [h[4].conj(), -h[0].conj(), h[12].conj(), -h[8].conj()],  [h[5].conj(), -h[1].conj(), h[13].conj(), -h[9].conj()]])
            
            
            n = np.array(n)
            Noise = [n[0], n[1], n[2], n[3], n[4].conj(), n[5].conj(), n[6].conj(), n[7].conj(), n[8].conj(), n[9].conj(), n[10].conj(), n[11].conj(), n[12], n[13], n[14], n[15]]
            Noise4by4_Uncoded = [n[0], n[1], n[2], n[3]]
            Noise4by2 =  [n[0], n[1], n[0].conj(), n[1].conj()]   
            
            rx_symbol_Uncoded =  np.add((np.dot(H4by4_Uncoded, (np.dot((1/sqrt(2)),symbol)))), Noise4by4_Uncoded) #recieved_symbol, y == rx_symbol
            rx_symbol =  np.add((np.dot(H, (np.dot((1/sqrt(2)),symbol)))), Noise) #recieved_symbol, y == rx_symbol

            rx_symbol4by2 =  np.add((np.dot(H4by2, (np.dot((1/sqrt(2)),symbol)))), Noise4by2) #recieved_symbol,
            
            #zero_forcing decoder generates the estimate of the transmitted matrix as: x: = pinv(H)*y
            Symbol_est = np.dot(np.linalg.pinv(H), rx_symbol)
            Symbol_est_Uncoded = np.dot(np.linalg.pinv(H4by4_Uncoded), rx_symbol_Uncoded)
            Symbol_est4by2 = np.dot(np.linalg.pinv(H4by2), rx_symbol4by2)
            
            #4by4
            det_symbol1 = 2*(Symbol_est[0].real >= 0) - 1
            det_symbol2 = 2*(Symbol_est[1].real >= 0) - 1
            det_symbol3 = 2*(Symbol_est[2].real >= 0) - 1
            det_symbol4 = 2*(Symbol_est[3].real >= 0) - 1
            
            #4by4_Uncoded
            det_symbol1_Uncoded = 2*(Symbol_est_Uncoded[0].real >= 0) - 1
            det_symbol2_Uncoded = 2*(Symbol_est_Uncoded[1].real >= 0) - 1
            det_symbol3_Uncoded = 2*(Symbol_est_Uncoded[2].real >= 0) - 1
            det_symbol4_Uncoded = 2*(Symbol_est_Uncoded[3].real >= 0) - 1
            
            #4by2          
            det_symbol1_4by2 = 2*(Symbol_est4by2[0].real >= 0) - 1
            det_symbol2_4by2 = 2*(Symbol_est4by2[1].real >= 0) - 1
            det_symbol3_4by2 = 2*(Symbol_est4by2[2].real >= 0) - 1
            det_symbol4_4by2 = 2*(Symbol_est4by2[3].real >= 0) - 1
            
            no_errors_Uncoded += 1*(tx_symbol1 != det_symbol1_Uncoded)+1*(tx_symbol2 != det_symbol2_Uncoded) +1*(tx_symbol3 != det_symbol3_Uncoded)+1*(tx_symbol4 != det_symbol4_Uncoded) 
            no_errors += 1*(tx_symbol1 != det_symbol1)+1*(tx_symbol2 != det_symbol2) +1*(tx_symbol3 != det_symbol3)+1*(tx_symbol4 != det_symbol4)  
            no_errors4by2 += 1*(tx_symbol1 != det_symbol1_4by2)+1*(tx_symbol2 != det_symbol2_4by2) +1*(tx_symbol3 != det_symbol3_4by2)+1*(tx_symbol4 != det_symbol4_4by2)   
            
            ber[k] = 1.0*no_errors/(l*NoOfSamples)
            ber4by4_Uncoded[k] = 1.0*no_errors_Uncoded/(l*NoOfSamples)
            ber_4by2[k] = 1.0*no_errors4by2/(l*NoOfSamples) #note: the expectation value of the bit error ratio.
        
        print ("EbNodB:", EbNodB)
        print ("Numbder of errors for 4by4:", no_errors)
        print ("Error probability 4by4:", ber[k]) #bit error ratio
        print ("Numbder of errors for 4by2:", no_errors4by2)
        print ("Error probability 4by2:", ber_4by2[k])
        print ("Numbder of errors for Uncoded:", no_errors_Uncoded)
        print ("Error probability Uncoded:", ber4by4_Uncoded[k])
        
Alamouti_4byX()       
