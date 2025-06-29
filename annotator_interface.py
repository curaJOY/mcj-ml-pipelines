import streamlit as st
import pandas as pd

# Your 50 sample posts — make sure this is defined at the top level!
SAMPLE_POSTS = [
    {"id": 1, "text": "Hope you have fun at the dance! 😊 (JK nobody will dance with you)"},
    {"id": 2, "text": "Bro you're literally killing me with these memes 😂💀"},
    {"id": 3, "text": "Nobody cares about your opinion, just stop talking."},
    {"id": 4, "text": "Great job on your presentation! Really inspiring stuff."},
    {"id": 5, "text": "You’re such a loser, can’t believe you even tried."},
    {"id": 6, "text": "Had the best day with friends! Feeling blessed 😄"},
    {"id": 7, "text": "Wow, you really mess things up every single time."},
    {"id": 8, "text": "Keep up the hard work, you’re making progress every day!"},
    {"id": 9, "text": "You’re pathetic, just disappear already."},
    {"id": 10, "text": "Thanks for the support, really appreciate it everyone!"},
    {"id": 11, "text": "That cut looks bad. Are you okay? Please talk to someone."},
    {"id": 12, "text": "You're literally the worst player on this team. Quit already."},
    {"id": 13, "text": "Can’t stop crying. Everything hurts."},
    {"id": 14, "text": "Why does she even try? No one likes her."},
    {"id": 15, "text": "You're amazing — don’t let anyone tell you otherwise."},
    {"id": 16, "text": "LMAO you're such a clown 🤡"},
    {"id": 17, "text": "I really need someone to talk to. This is too much."},
    {"id": 18, "text": "Omg you're so dumb it actually hurts."},
    {"id": 19, "text": "Proud of you for pushing through, seriously 👏"},
    {"id": 20, "text": "She should just disappear already tbh."},
    {"id": 21, "text": "Big W! You smashed that test! 🎉"},
    {"id": 22, "text": "You're not even worth hating, that’s how irrelevant you are."},
    {"id": 23, "text": "Is it bad that I just want to stop existing for a bit?"},
    {"id": 24, "text": "You're an attention seeker. Nobody cares about your 'anxiety'."},
    {"id": 25, "text": "Thanks for checking in, it means a lot ❤️"},
    {"id": 26, "text": "Bet she cried again. Crybaby."},
    {"id": 27, "text": "Life feels empty sometimes."},
    {"id": 28, "text": "You're doing your best, and that’s enough."},
    {"id": 29, "text": "He’s like a virus in every group chat."},
    {"id": 30, "text": "That post made my day, thank you for sharing it!"},
    {"id": 31, "text": "Your makeup looks like a failed science experiment 💀"},
    {"id": 32, "text": "Please don’t hurt yourself. You matter."},
    {"id": 33, "text": "Did you even read what you posted? So cringe."},
    {"id": 34, "text": "We’re here for you. Don’t isolate yourself."},
    {"id": 35, "text": "Bro just log off if you can't handle the heat 😂"},
    {"id": 36, "text": "No one invited you for a reason."},
    {"id": 37, "text": "You seriously need help. And not the good kind."},
    {"id": 38, "text": "You're stronger than you think 💪"},
    {"id": 39, "text": "She's acting depressed for clout again 🙄"},
    {"id": 40, "text": "Hey, I saw your story — you okay?"},
    {"id": 41, "text": "I wish I could just sleep forever."},
    {"id": 42, "text": "This community is so supportive. Love you all."},
    {"id": 43, "text": "Honestly I wouldn't care if something happened to him."},
    {"id": 44, "text": "You did amazing today, don't let one mistake ruin it."},
    {"id": 45, "text": "Every time you speak, my brain loses IQ points."},
    {"id": 46, "text": "I feel invisible all the time."},
    {"id": 47, "text": "What a kind comment! You made my day."},
    {"id": 48, "text": "Maybe the world would be better without me."},
    {"id": 49, "text": "I’m so proud of how far you’ve come."},
    {"id": 50, "text": "Can someone tell her to shut up already?"}
    {"id": 51, "text": "I swear I'll kill myself today if this pain doesn't go away"}
    {"id": 52, "text": "Kill yourself and go deeper next time"}
    {"id": 53, "text": "Good job! Pink looks good on you"}
    {"id": 54, "text": "I love that we have the same favorite artist!"}
    {"id": 55, "text": "EWW. What is it doing?"}
    {"id": 56, "text": "That looks so ridiculous on you, don't wear it again"}
    {"id": 57, "text": "I feel so numb everytime"}
    {"id": 58, "text": "Shut your mouth, you stupid turd"}
    {"id": 59, "text": "I hope I'am no longer here by next month"}
    {"id": 60, "text": "Can't wait to watch the game tomorrow"}
new_samples = [
    # Bullying phrases
    {"id": 61, "text": "You're such a loser, nobody likes you.", "bullying": "Yes"},
    {"id": 62, "text": "Why don’t you just go away?", "bullying": "Yes"},
    {"id": 63, "text": "Nobody wants you here.", "bullying": "Yes"},
    {"id": 64, "text": "You’re so dumb, it’s embarrassing.", "bullying": "Yes"},
    {"id": 65, "text": "You’re a failure at everything you do.", "bullying": "Yes"},
    {"id": 66, "text": "Go cry somewhere else, we don’t care.", "bullying": "Yes"},
    {"id": 67, "text": "You’re ugly and no one will ever love you.", "bullying": "Yes"},
    {"id": 68, "text": "Stop being such a pathetic joke.", "bullying": "Yes"},
    {"id": 69, "text": "You’re worthless, just disappear.", "bullying": "Yes"},
    {"id": 70, "text": "Everyone thinks you’re annoying.", "bullying": "Yes"},
    {"id": 71, "text": "You’ll never be good enough.", "bullying": "Yes"},
    {"id": 72, "text": "You’re such a freak.", "bullying": "Yes"},
    {"id": 73, "text": "No one wants to be your friend.", "bullying": "Yes"},
    {"id": 74, "text": "You’re a complete embarrassment.", "bullying": "Yes"},
    {"id": 75, "text": "Why do you even bother trying?", "bullying": "Yes"},7
    # Not bullying phrases
    {"id": 76, "text": "Great job on your test!", "bullying": "No"},
    {"id": 77, "text": "I hope you have a wonderful day.", "bullying": "No"},
    {"id": 78, "text": "Thanks for helping me out.", "bullying": "No"},
    {"id": 79, "text": "You did amazing today.", "bullying": "No"},
    {"id": 80, "text": "Keep up the fantastic work!", "bullying": "No"},
    {"id": 81, "text": "I’m proud of you.", "bullying": "No"},
    {"id": 82, "text": "You’re doing really well.", "bullying": "No"},
    {"id": 83, "text": "Let me know if you need help.", "bullying": "No"},
    {"id": 84, "text": "That was a really kind thing to do.", "bullying": "No"},
    {"id": 85, "text": "You’re a good friend.", "bullying": "No"},
    {"id": 86, "text": "Don’t give up, you’re almost there!", "bullying": "No"},
    {"id": 87, "text": "I believe in you.", "bullying": "No"},
    {"id": 88, "text": "You have so much potential.", "bullying": "No"},
    {"id": 89, "text": "It’s okay to make mistakes, we learn from them.", "bullying": "No"},
    {"id": 90, "text": "Thanks for sharing that with me.", "bullying": "No"},
]
[
  {"id": 91, "text": "You are stupid", "bullying": "Yes"},
  {"id": 92, "text": "You are a fool", "bullying": "Yes"},
  {"id": 93, "text": "You’re an idiot", "bullying": "Yes"},
  {"id": 94, "text": "Nobody likes you", "bullying": "Yes"},
  {"id": 95, "text": "You’re ugly", "bullying": "Yes"},
  {"id": 96, "text": "You’re worthless", "bullying": "Yes"},
  {"id": 97, "text": "Go kill yourself", "bullying": "Yes"},
  {"id": 98, "text": "You’re trash", "bullying": "Yes"},
  {"id": 99, "text": "You can’t do anything right", "bullying": "Yes"},
  {"id": 100, "text": "You’re a loser", "bullying": "Yes"},
  {"id": 101, "text": "You’re annoying", "bullying": "Yes"},
  {"id": 102, "text": "You’re a joke", "bullying": "Yes"},
  {"id": 103, "text": "No one wants you here", "bullying": "Yes"},
  {"id": 104, "text": "You’re disgusting", "bullying": "Yes"},
  {"id": 105, "text": "Stop talking, nobody cares", "bullying": "Yes"},

  {"id": 106, "text": "You’re so silly", "bullying": "No"},
  {"id": 107, "text": "That’s dumb, but funny", "bullying": "No"},
  {"id": 108, "text": "You’re joking, right?", "bullying": "No"},
  {"id": 109, "text": "Chill, it’s not a big deal", "bullying": "No"},
  {"id": 110, "text": "You’re crazy", "bullying": "No"},
  {"id": 111, "text": "Don’t be mad, it’s just a joke", "bullying": "No"},
  {"id": 112, "text": "Everyone makes mistakes", "bullying": "No"},
  {"id": 113, "text": "You’re just joking around", "bullying": "No"},
  {"id": 114, "text": "It’s all good, no worries", "bullying": "No"},
  {"id": 115, "text": "You’re overthinking it", "bullying": "No"}
]
[
  {"id": 116, "text": "You’re so dumb", "bullying": "Yes"},
  {"id": 117, "text": "You’re pathetic", "bullying": "Yes"},
  {"id": 118, "text": "Nobody cares about you", "bullying": "Yes"},
  {"id": 119, "text": "You’re disgusting", "bullying": "Yes"},
  {"id": 120, "text": "Go away, loser", "bullying": "Yes"},
  {"id": 121, "text": "You’re a waste of space", "bullying": "Yes"},
  {"id": 122, "text": "You ruin everything", "bullying": "Yes"},
  {"id": 123, "text": "You’re so annoying", "bullying": "Yes"},
  {"id": 124, "text": "Why are you even here?", "bullying": "Yes"},
  {"id": 125, "text": "You’re ugly inside and out", "bullying": "Yes"},
  {"id": 126, "text": "You have no friends because you suck", "bullying": "Yes"},
  {"id": 127, "text": "You’re the worst", "bullying": "Yes"},
  {"id": 128, "text": "I hope you fail", "bullying": "Yes"},
  {"id": 129, "text": "You’re such a disappointment", "bullying": "Yes"},
  {"id": 130, "text": "Everyone hates you", "bullying": "Yes"},
  {"id": 131, "text": "You don’t belong here", "bullying": "Yes"},
  {"id": 132, "text": "You’re an embarrassment", "bullying": "Yes"},
  {"id": 133, "text": "Stop trying, you’re bad at everything", "bullying": "Yes"},
  {"id": 134, "text": "No one will miss you", "bullying": "Yes"},
  {"id": 135, "text": "You’re such a failure", "bullying": "Yes"}
]
[
  {"id": 136, "text": "You’re being dramatic", "bullying": "No"},
  {"id": 137, "text": "Calm down, it’s not that serious", "bullying": "No"},
  {"id": 138, "text": "You’re so funny", "bullying": "No"},
  {"id": 139, "text": "That was a dumb mistake", "bullying": "No"},
  {"id": 140, "text": "You’re acting weird", "bullying": "No"},
  {"id": 141, "text": "Don’t be so sensitive", "bullying": "No"},
  {"id": 142, "text": "You’re overreacting", "bullying": "No"},
  {"id": 143, "text": "I don’t agree with you", "bullying": "No"},
  {"id": 144, "text": "You’re a bit annoying sometimes", "bullying": "No"},
  {"id": 145, "text": "That’s a stupid idea", "bullying": "No"},
  {"id": 146, "text": "You should lighten up", "bullying": "No"},
  {"id": 147, "text": "It’s just a joke, don’t take it seriously", "bullying": "No"},
  {"id": 148, "text": "You’re kind of extra today", "bullying": "No"},
  {"id": 149, "text": "You need to chill", "bullying": "No"},
  {"id": 150, "text": "That’s a silly thing to say", "bullying": "No"},
  {"id": 151, "text": "You’re weird, but I like it", "bullying": "No"},
  {"id": 152, "text": "Don’t sweat the small stuff", "bullying": "No"},
  {"id": 153, "text": "You’re acting childish", "bullying": "No"},
  {"id": 154, "text": "That’s an interesting take", "bullying": "No"},
  {"id": 155, "text": "You’re joking, right?", "bullying": "No"}
]


]

def create_annotation_interface():
    st.title("Cyberbullying Annotation Tool")

    # Load existing annotations if file exists
    try:
        annotated_df = pd.read_csv("annotations.csv")
        annotated_ids = set(annotated_df["id"])
    except FileNotFoundError:
        annotated_df = pd.DataFrame()
        annotated_ids = set()

    # Find next post to annotate
    next_post = None
    for post in SAMPLE_POSTS:
        if post["id"] not in annotated_ids:
            next_post = post
            break

    if next_post is None:
        st.success("✅ All posts have been annotated!")
        return

    st.subheader("Post")
    st.write(f"**Text:** {next_post['text']}")

    st.markdown("---")
    st.subheader("Annotation")

    bullying = st.radio("Is this post bullying?", ["Yes", "No"])
    self_harm = st.radio("Does this post indicate self-harm?", ["Yes", "No"])
    severity = st.selectbox("How severe is this post?", ["Low", "Medium", "High"])
    confidence = st.slider("How confident are you?", 0.0, 1.0, 0.8)
    notes = st.text_area("Notes / edge cases (optional)")

    if st.button("Submit Annotation"):
        new_annotation = {
            "id": next_post["id"],
            "bullying": bullying,
            "self_harm": self_harm,
            "severity": severity,
            "confidence": confidence,
            "notes": notes
        }

        annotated_df = pd.concat([annotated_df, pd.DataFrame([new_annotation])], ignore_index=True)
        annotated_df.to_csv("annotations.csv", index=False)
        st.success("✅ Annotation saved! Please refresh or rerun to see the next post.")

if __name__ == "__main__":
    create_annotation_interface()
