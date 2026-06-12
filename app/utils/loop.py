import asyncio

event_loop: asyncio.AbstractEventLoop = None

def set_loop(loop: asyncio.AbstractEventLoop):
    global event_loop
    event_loop = loop

def get_loop() -> asyncio.AbstractEventLoop:
    return event_loop