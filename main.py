from tinyros.build.output.client_library.python.tinyros_hello.msg import TinyrosHello

msg = TinyrosHello()
msg.data = "Hello from Python!"
print("✅ 成功匯入並建立訊息：", msg)

