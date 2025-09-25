import os
import logging
import discord
from dotenv import load_dotenv
from discord.ext import commands
from utils import extract_chunks
from price_fetcher import fetch_token_price
import asyncio

TOKEN_MAP = {
    "bera":"berachain",
    "bgt":"berachain-governance-token",
    "ibgt":"infrared-bgt",
    "lbgt":"liquid-bgt"
}

from rag_model import RAGModel

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DOCS_PATH = os.getenv("DOCS_PATH", "/home/pawreador/Paw-Reador/docs") #in the hosted vps

# Initialize RAG model (indexing at startup)
rag = RAGModel(docs_path=DOCS_PATH, context_window=5, cache_size=200)

# Set up Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

user_threads = {}

@bot.event
async def on_ready():
    logging.info(f"Bot connected as {bot.user}")

@bot.command(name="ask")
async def ask(ctx, *, question: str):
    """Handle !ask command: use RAG to answer user questions."""
    try:
        async with ctx.typing():
            result = rag.query_async(question, str(ctx.author.id), k=5)
            # Extract text from AIMessage or string
            if result["rate_limited"]:
                await ctx.send(result['rate_limited'])
                return

            response_text = result['response']

            indicators = []
            if result['cached']:
                indicators.append("💾")
            if result.get('error'):
                indicators.append("⚠️")

            prefix = " ".join(indicators) + " " if indicators else ""

            await send_long_messages(ctx, f"{prefix}{response_text}")
            
    except Exception as e:
        logging.exception("Error in ask command")
        await ctx.send("Oof, something went wrong on our end! We're fixing it rn 🔧")

@bot.command(name="chat")
async def start_personal_chat(ctx):
    """Start a personal thread for contextual conversations."""
    try:
        user_id = str(ctx.author.id)
        
        # Check if user already has an active thread
        if user_id in user_threads:
            try:
                existing_thread = bot.get_channel(user_threads[user_id])
                if existing_thread and not existing_thread.archived:
                    await ctx.send(f"You already have an active chat thread! 🧵 {existing_thread.mention}")
                    return
                else:
                    # Thread was archived or deleted, remove from tracking
                    del user_threads[user_id]
            except:
                # Thread doesn't exist anymore
                del user_threads[user_id]
        
        # Create new thread
        thread_name = f"🐻 {ctx.author.display_name}'s BeraFarm Chat"
        thread = await ctx.message.create_thread(
            name=thread_name,
            auto_archive_duration=1440  # 24 hours
        )
        
        # Track the thread
        user_threads[user_id] = thread.id
        
        # Welcome message in thread
        welcome_msg = (
            f"gm {ctx.author.mention}! 🚀\n\n"
            "Welcome to your personal BeraFarm chat! Here I'll remember our conversation "
            "and give you more contextual responses.\n\n"
            "Just ask me anything about BeraFarm, Berachain, or DeFi stuff - no need for `!ask` here!\n\n"
            "**Commands you can use:**\n"
            "• Just type your question directly\n"
            "• `!stats` - See chat stats\n"
            "• `!clear` - Clear our chat history\n"
            "• `!price <token>` - Check token prices\n\n"
            "*This thread will auto-archive after 24h of inactivity* ⏰"
        )
        
        await thread.send(welcome_msg)
        await ctx.send(f"Started your personal chat thread! 🧵 {thread.mention}")
        
    except Exception as e:
        logging.exception("Error creating personal chat thread")
        await ctx.send("Couldn't start your chat thread rn, try again in a sec!")

async def on_message(message):
    # Ignore bot messages
    if message.author == bot.user:
        return
    
    # Handle thread messages (auto-chat without !ask)
    if isinstance(message.channel, discord.Thread):
        user_id = str(message.author.id)
        
        # Check if this is a tracked user thread
        if user_id in user_threads and user_threads[user_id] == message.channel.id:
            # Don't process commands in threads (they should use regular chat)
            if message.content.startswith('!'):
                if message.content.startswith('!clear'):
                    await clear_user_history(message)
                    return
                elif message.content.startswith('!stats'):
                    await show_user_stats(message)
                    return
                elif message.content.startswith('!price'):
                    # Process price command
                    await bot.process_commands(message)
                    return
                else:
                    await message.channel.send("Use commands in the main channel, not in your personal thread! Just ask questions here directly 😊")
                    return
            
            # Process as RAG query
            try:
                async with message.channel.typing():
                    result = await rag.query_async(message.content, user_id, k=5)
                    
                    if result['rate_limited']:
                        await message.channel.send(result['response'])
                        return
                    
                    response_text = result['response']
                    
                    # Add subtle indicators for thread
                    prefix = "💾 " if result['cached'] else ""
                    
                    await send_long_message(message.channel, f"{prefix}{response_text}")
                    
            except Exception as e:
                logging.exception("Error in thread message processing")
                await message.channel.send("Oof, hit a snag there! Try asking again 🔧")
            
            return  # Don't process as command
    
    # Process regular commands
    await bot.process_commands(message)

async def clear_user_history(message):
    """Clear user's chat history"""
    user_id = str(message.author.id)
    rag.clear_user_history(user_id)
    await message.channel.send("Cleared your chat history! Starting fresh 🧹✨")

async def show_user_stats(message):
    """Show user-specific stats"""
    try:
        stats = rag.get_stats()
        user_id = str(message.author.id)
        
        # Check if user has history
        has_history = user_id in rag.chat_histories
        history_count = len(rag.chat_histories.get(user_id, []))
        
        embed = discord.Embed(
            title="📊 Your BeraFarm Chat Stats",
            color=0x8B4513,  # Brown for bear theme
            description="Here's what we've been cooking together!"
        )
        
        embed.add_field(
            name="Your Stats", 
            value=f"Messages remembered: {history_count}\nHave history: {'✅' if has_history else '❌'}", 
            inline=True
        )
        
        embed.add_field(
            name="Global Stats", 
            value=f"Total users: {stats['total_users']}\nActive now: {stats['active_users']}\nDocs loaded: {stats['total_documents']}", 
            inline=True
        )
        
        embed.set_footer(text="Use !clear to reset your history")
        
        await message.channel.send(embed=embed)
        
    except Exception as e:
        logging.exception("Error showing stats")
        await message.channel.send("Couldn't grab stats right now, we're fixing it! 🔧")

async def send_long_message(channel, text, max_len=2000):
    """Send long messages by splitting them properly"""
    if len(text) <= max_len:
        await channel.send(text)
    else:
        # Split by sentences/paragraphs first, then by length
        parts = text.split('\n\n')
        current_message = ""
        
        for part in parts:
            if len(current_message) + len(part) + 2 <= max_len:
                current_message += part + "\n\n"
            else:
                if current_message:
                    await channel.send(current_message.strip())
                    current_message = part + "\n\n"
                else:
                    # Part itself is too long, force split
                    for i in range(0, len(part), max_len):
                        await channel.send(part[i:i+max_len])
                    current_message = ""
        
        if current_message:
            await channel.send(current_message.strip())

@bot.command(name="ping")
async def ping(ctx):
    await ctx.send(f"Pong! Latency: {round(bot.latency*1000)}ms")
    
@bot.command(name="price")
async def price(ctx, *, token_name: str):
    """Fetches the price for a given token."""
    token_id = TOKEN_MAP.get(token_name.lower())
    if not token_id:
        await ctx.send(f"Sorry, I don't know the token '{token_name}'. Try one of: {', '.join(TOKEN_MAP.keys())}")
        return
    
    try:
        async with ctx.typing():
            prices = await fetch_token_price(token_id)
            if not prices:
                await ctx.send("Sorry, I couldn't fetch the price data right now.")
                return
            
            change = calculate_change(prices)
            response = (f"**{token_name.upper()} Price**\n"
                        f"Current: `${prices['current']:.4f}`\n"
                        f"Yesterday: `${prices['yesterday']:.4f}`\n"
                        f"24h Change: `{change:.2f}%`")
            await ctx.send(response)
    except Exception as e:
        logging.exception(f"Error fetching price for {token_name}")
        await ctx.send("Sorry, an error occurred while fetching the price.")

#Helper functions
def calculate_change(prices: dict) -> float:
    if prices['yesterday'] == 0:
        return float('inf') if prices['current'] > 0 else 0.0
    return ((prices['current'] - prices['yesterday']) / prices['yesterday']) * 100

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot.run(TOKEN)