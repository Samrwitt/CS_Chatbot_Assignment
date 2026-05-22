from app import create_app

app = create_app()

if __name__ == '__main__':
    # Run the application on port 5005
    app.run(debug=True, port=5005)
