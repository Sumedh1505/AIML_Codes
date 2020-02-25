# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:08:52 2020

@author: 205557
"""


from cryptography.fernet import Fernet
import base64
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class ML_encrypt():
    def _init_(self):
        pass
    def ecrypt_random(self,filename):
        key= Fernet.generate_key()
        file= open(filename,'wb')
        file.write(key)
        file.close()
    def encrypt_from_password(self, filename, password='spiml',salt= b'salt'):
        password_provided = password # This is input in the form of a string
        password = password_provided.encode() # Convert to type bytes
        salt = salt # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
                    )
        key = base64.urlsafe_b64encode(kdf.derive(password))   
        file= open(filename,'wb')
        file.write(key)
        file.close()
    def read_encryption(self, filename):
        file= open(filename,'rb')
        key= file.read()
        file.close()
        return(key)
    def encrypt_message(self, text, filename):
        key= self.read_encryption(filename)
        f = Fernet(key)
        if type(text)== list:
            encrypted_list= []
            for content in text:
                message = content.encode()
                encrypted = f.encrypt(message)
                encrypted_list.append(encrypted)
            return encrypted_list
        else:
            message = text.encode()
            encrypted = f.encrypt(message)
            return encrypted
           
    def decrypt_message(self, text, filename):
        key= self.read_encryption(filename)
        f = Fernet(key)
        if type(text)== list:
            decrypted_list= []
            for content in text:
                content= content.encode()
                decrypted = f.decrypt(content)
                decrypted_list.append(decrypted.decode().strip())
            return decrypted_list
        else:
            decrypted = f.decrypt(text)
            return decrypted     
        

        
