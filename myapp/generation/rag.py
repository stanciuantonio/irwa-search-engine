import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
You are an expert fashion product advisor. Your task is to help users find the BEST product from retrieved results.

CRITICAL INSTRUCTIONS:
1. Analyze ALL retrieved products carefully
2. Consider: price, rating, discount, availability, quality fit
3. Recommend THE SINGLE BEST product
4. Explain why this is the best choice (specific reasons)
5. Optionally mention ONE alternative if relevant
6. If NO product is suitable, respond with ONLY: "No good products match your request based on current results."

USER REQUEST:
{user_query}

RETRIEVED PRODUCTS (Top Results):
{enriched_results}

YOUR RESPONSE:
- **Best Product:** [PID and Name]
- **Why This One:** [Clear explanation with specific attributes - price, quality, ratings, fit]
- **Alternative (if any):** [Optional - another good option]

Remember: Be concise, specific, and helpful. Focus on what matters to the user.
    """

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 5) -> str:
        """
        Generate a response using the retrieved search results.
        Returns:
            str: AI-generated recommendation or explanation
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        try:
            # Prepare enriched product information
            if not retrieved_results:
                return "No products found to summarize."

            enriched_results = self._format_enriched_results(retrieved_results[:top_N])

            # Build the prompt
            prompt = self.PROMPT_TEMPLATE.format(
                user_query=user_query,
                enriched_results=enriched_results
            )

            # Call Groq API
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return generation

        except Exception as e:
            print(f"[RAG] Error during generation: {e}")
            return DEFAULT_ANSWER

    def _format_enriched_results(self, results: list) -> str:
        """
        Format search results with enriched product information for the LLM.

        Args:
            results: List of Document objects

        Returns:
            str: Formatted product information
        """
        formatted = []

        for idx, doc in enumerate(results, 1):
            price_info = f"₹{doc.selling_price:.2f}" if doc.selling_price else "N/A"
            discount_info = f"{int(doc.discount)}% OFF" if doc.discount else "No discount"
            rating_info = f"{doc.average_rating}/5" if doc.average_rating else "No rating"
            stock_info = "OUT OF STOCK" if doc.out_of_stock else "IN STOCK"

            # Format product details if available
            details_str = ""
            if doc.product_details:
                details_list = [f"{k}: {v}" for k, v in list(doc.product_details.items())[:3]]
                details_str = f"\n   Specs: {', '.join(details_list)}"

            product_info = f"""
                {idx}. [{doc.pid}] {doc.title}
                Brand: {doc.brand or 'N/A'} | Category: {doc.category or 'N/A'}
                Price: {price_info} ({discount_info})
                Rating: {rating_info} | Status: {stock_info}
                Description: {doc.description[:150] if doc.description else 'N/A'}...{details_str}
                """
            formatted.append(product_info)

        return "\n".join(formatted)
