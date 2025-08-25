import os
import redis

redis_url = os.getenv("REDIS_URL")

if not redis_url:
    print("❌ ERROR: The REDIS_URL environment variable is not set.")
else:
    print(f"Attempting to connect to: {redis_url}")
    try:
        # Using the same robust parameters
        client = redis.from_url(
            redis_url,
            health_check_interval=30,
            socket_connect_timeout=5,
            socket_keepalive=True,
            decode_responses=True
        )
        
        # Ping the server to check the connection
        response = client.ping()
        
        if response:
            print("✅ Successfully connected to Redis and received a PONG!")
        else:
            print("⚠️ Connected to Redis but the PING command failed.")

    except redis.exceptions.AuthenticationError:
        print("❌ CONNECTION FAILED: Authentication error. Check your password.")
    except redis.exceptions.ConnectionError as e:
        print(f"❌ CONNECTION FAILED: A connection error occurred. Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")