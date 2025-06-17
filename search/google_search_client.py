from langchain_core.tools import tool
from requests import request

class GoogleSearchClient:
    GOOGLE_SEARCH_API_KEY = "AIzaSyB6jRXhI-GmvAiiMy0W2zeEFPxAZCmM5eQ"
    GOOGLE_SEARCH_CX = "90cab94bff40a402f"
    GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __str__(self):
        return "Google Search Client with API Key: " + self.GOOGLE_SEARCH_API_KEY

    @staticmethod
    @tool("google_search_tool", return_direct=False)
    def search(query: str) -> str:
        """
        A tool that performs a Google search and returns the results.

        Args:
            query: The search query to send to Google

        Returns:
            str: JSON string containing search results or error message
        """
        params = {
            'key': GoogleSearchClient.GOOGLE_SEARCH_API_KEY,
            'cx': GoogleSearchClient.GOOGLE_SEARCH_CX,
            'q': query,
        }
        print("Searching Google for:", query)
        response = request("GET", GoogleSearchClient.GOOGLE_SEARCH_ENDPOINT, params=params)

        try:
            response.raise_for_status()
            return str(response.json())
        except Exception as e:
            error_msg = f"Error occurred: {e}"
            print(error_msg)
            return error_msg


# if __name__ == '__main__':
#