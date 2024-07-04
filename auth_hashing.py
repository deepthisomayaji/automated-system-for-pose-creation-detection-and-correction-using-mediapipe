from streamlit_authenticator.utilities.hasher import Hasher

hashed_passwords = Hasher(['admin']).generate()
print(hashed_passwords)
