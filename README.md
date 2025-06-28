# Table of Contents
- [Table of Contents](#table-of-contents)
- [Project overview](#project-overview)
- [Data Validation, Findings \& Assumptions](#data-validation-findings--assumptions)
  - [1. Data cleaning](#1-data-cleaning)
  - [2.Key Customer Segments](#2key-customer-segments)
  - [3. Informational Offers and Completion rates](#3-informational-offers-and-completion-rates)
  - [4. Offer Driven Sales Calculation](#4-offer-driven-sales-calculation)
- [Deep Insights](#deep-insights)
  - [1. How many reward offers were completed? Which offers had the highest completion rate?](#1-how-many-reward-offers-were-completed-which-offers-had-the-highest-completion-rate)
  - [2. How many informational offers were followed by transactions?](#2-how-many-informational-offers-were-followed-by-transactions)
  - [3. How are customer demographics distributed?](#3-how-are-customer-demographics-distributed)
  - [4. Are there any demographic patterns in offer completion?](#4-are-there-any-demographic-patterns-in-offer-completion)
- [Customer Segments Analysis](#customer-segments-analysis)
- [Offer Journey](#offer-journey)
- [Turning Insight into Action from Final Takeaways](#turning-insight-into-action-from-final-takeaways)
- [Final Recommendations](#final-recommendations)




# Project overview
[Back to Table of Contents](#table-of-contents)

_Café Rewards_ program just run a test by sending different combinations of promotional offers to existing rewards members. Now that the **30-day period** for the test has concluded, the task is to **identify key customer segments and develop a data-driven strategy for future promotional messaging & targeting**. The results should be summarized in a report that will be presented to the Chief Marketing Officer (CMO).

The data consists of a table with **17k** unique existing rewards members including their age, income, gender, and date of membership; and another table capturing all events during the **30-day period**, such as when offers were sent, viewed and completed, along with all customer transactions with the cafe.

The data used in this project is available [here](Data)

# Data Validation, Findings & Assumptions
[Back to Table of Contents](#table-of-contents)
## 1. Data cleaning
The data cleaning process was relatively straightforward and therefore not too time consuming. The main challenge was deciding how to handle  a group of **2,175 (12.8%)** customers  with missing values, particularly those with an age recorded as 118 were identified with null value for Gender & Income. For these cases   I chose to remove these entries from the dataset entirely because it didn't have any impact on findings.

I created bins for age and income to group customers together. After some research, I defined the following age groups:
- **Age Groups**: Young Adults (18–24), Early Career (25–34), Young Families (35–44), Mature Professionals (45–54), Pre-Retirement (55–64), Retirees (65+)
    
- **Income Brackets**: Lower-Middle (30–50k), Middle (50–75k), Upper-Middle (75–100k), Affluent (100–120k)
    
These groupings provided a solid overview of customer demographics and revenue distribution.

## 2.Key Customer Segments
I performed  RFM segmentation and clustering on the  dataset to identify distinct customer groups for targeted marketing. After importing and filtering transaction data, Recency, Frequency, and Monetary metrics were calculated per customer. Outliers were identified using boxplots and the IQR method, then removed to ensure clean clustering. StandardScaler was applied before running K-Means, with `k=4` selected using the Elbow and Silhouette methods. Clusters were profiled and labeled (e.g., Champions, Slipping Away) based on RFM behavior. Outliers were separately clustered and analyzed for metric-specific extremities. The final segmentation combined core clusters and outliers, which were mapped back to the full event dataset for integrated analysis. 

Here is a detailed [*Technical Documentation of  RFM Segmentation and Clustering Thought Process*](Scripts)

Here is a snippet of K-Means and Silhoutte score

![alt text](<Detailed RFM Customer Segmentation/Images/k_means_and_silhoutte.png>)

Here is a combination of a bar plot and line plot to summarize customer clusters:

![alt text](<Detailed RFM Customer Segmentation/Images/Customer_distribution.png>)
- **Bar Plot:** Shows the number of customers in each cluster (segment), helping you see which segments are largest or smallest.
- **Line Plot:** Overlays the average values of Recency, Frequency, and Monetary Value (per $100) for each cluster, illustrating how customer behavior differs by segment.

**Interpretation:**
- Taller bars indicate more customers in that segment.
- The line plot reveals which clusters have higher spenders, more frequent buyers, or more recent activity.
- Together, these visuals help identify high-value segments, potential targets for retention, and areas for marketing focus.

## 3. Informational Offers and Completion rates

To calculate the completion rate, informational offers are not considered. It was assumed that these do not apply to the reward criteria since they are only for informational purposes for the customer. So completion rate only consider bogo + discount offers. For the general calculation of % Viewed, informational offer was included.

## 4. Offer Driven Sales Calculation

It was calculated based on the sales of transactions attributed to a completed offer. Some notes to keep in mind:

There were cases of transactions with more than 1 offer completed related. This is ok, but when distributing the amount by type of offer the result will not be the exact sum of both (because they coincide in some transactions).

Of the 33,579 completed offer events, there  were  instances where customers completed nearly every offer before viewing it

# Deep Insights

[Back to Table of Contents](#table-of-contents)
## 1. How many reward offers were completed? Which offers had the highest completion rate?

**Business Metric Insight:**  
- Both BOGO and Discount offers achieved **comparable completion volumes**, **BOGO offers were completed faster** (average of 2.18 days vs. 2.99).  This suggests BOGO deals may trigger **quicker action**, whereas discounts might require **more consideration or timing**. From a campaign planning view, **BOGO offers are more time-sensitive** but can drive **faster conversions**  
- The offer with the highest completion rate is `fafdcd668e3743c1bb461111dcafc2a4` at **85.15%**, followed closely by `2298d6c36e964ae4a3e7e9706d1fb8c2` (**84.40%**).  Two offers `3f207df678b143eea3cee63160fa8bed` and `5a8bc65990b245e5a138643cd4eb9837` had **zero completions** despite being received over 6,600 times each, indicating low engagement or relevance.  The overall average completion rate across all offers is **55.38%**, suggesting that more than half of issued offers lead to completions
- Discount offers have a slightly **higher completion rate (71.51%)** compared to BOGO offers (66.91%).  Both types perform strongly, but **discounts edge out BOGO** in effectiveness.  This suggests customers may prefer **immediate savings** over bundled deals.

## 2. How many informational offers were followed by transactions?
**Business Metric Insight:**
- **2,019 transactions were influenced by informational offers**, suggesting that **brand messaging and awareness campaigns** still drive measurable customer action  especially when distributed across **multi-channel strategies** (email, mobile, social, web). 

- **95%** of attributed transactions (**36,825 out of 38,844**) came from **offers with a completion mechanism** (BOGO, Discount). Only **5%** were influenced by **Informational** offers, which don’t require explicit completion but still drove purchases.

 - While informational offers have a lower transaction count, they still show **measurable influence**. However, BOGO and Discount offers are far more effective in driving **direct, attributed purchases**.
## 3. How are customer demographics distributed?
**Key Insights**

- The majority of customers fall under the **Middle Income tier** (almost **45%**), followed by **Lower Middle** and **Upper Middle**.
- In terms of life stages: **Retirees** make up the largest group at **28%**, indicating an older customer base. **Pre-Retirement** and **Mature Professionals** also represent significant portions, showing the brand’s strong appeal among **older working-age adults**.
- **Younger segments** : **Young Adults** and **Early Career Professionals**  account for **just 16.3%**, suggesting either a smaller audience or an opportunity for targeted marketing.
## 4. Are there any demographic patterns in offer completion?

**Key Patterns and Insights**

- **Middle Income customers** had the **highest number of completed offers** (16,004), accounting for **~43%** of all completions.
    
- **Upper Middle Income** and **Lower Middle Income** groups followed showing strong engagement in both mid and slightly higher income bands.
    
- Among **life stages**, **Retirees** and **Pre-Retirement** individuals had a notably high completion count , **older segments appear more responsive** to offers.
    
- **Young Adults** had the **lowest completion rate** (806 completions), suggesting **lower engagement or conversion** among this group.
    
- **Mature Professionals** consistently appear across multiple categories, showing **stable participation across age/income segments**.
    
**Business Takeaway**

- **Targeting Middle Income and Retirees/Pre-Retirement** segments may yield the **highest offer engagement**.
    
- **Younger users** (Early Career, Young Adults) may require **more personalized or digital-centric strategies** to improve conversion.
    
- Demographic segmentation can play a vital role in **offer design and distribution channels**


#  Customer Segments Analysis
[Back to Table of Contents](#table-of-contents)
- Among all customer segments, **Frequent Shoppers** stood out as the most responsive group, contributing to **33.2%** of completed offer attributions , that’s **4,287** out of **12,440**. Despite their enthusiasm, they take the longest to act, with an average of **13.29 days** to complete a transaction. This delay may reflect a longer decision-making process or a comfort with waiting for the perfect moment to make their move.

- Meanwhile, both **First-Time Buyers** and **Sleeping Buyers** showed encouraging signs of engagement, especially when it came to completed offers. These two groups , new and reactivating customers  are gaining traction. **First-Time Buyers** respond the fastest, completing transactions in just **8.5 days**, a promising sign that initial engagement strategies are working well. In contrast, **Sleeping Buyers** take over **11.5 days**, revealing a more passive, perhaps cautious, pattern of behavior.

- **High Spenders** and **VIP Customers** may not dominate in volume, but their transactions likely carry greater value. They’re a strategic segment , smaller in size, but rich in potential. Interestingly, they tend to act **faster than average**, showing a decisiveness or responsiveness that may be driven by targeted high-value incentives.

- Then there are the **Win-Back Targets** , customers with **very low interaction** overall. Their lack of engagement highlights the need for more assertive or personalized reactivation efforts. Yet, despite their low activity, when they do respond, they act **faster than average**, hinting that the right message could unlock their potential.
- **Loyal Core** shows steady engagement with consistent completions and moderate transaction time (**11.5 days**). Their behavior suggests reliability — they respond well to loyalty-focused campaigns and ongoing relationship nurturing.
- **Potential Upgraders** display promising signs of increased engagement, averaging **7.58 days** to transaction, which is faster than average. With the right incentives, this group could be encouraged to grow spend and transition into higher-value segments

- Looking at **informational offers**, these had the greatest impact on a few key groups: **First-Time Buyers** (658), **Sleeping Buyers** (299), and **Frequent Shoppers** (489). It’s clear that informational content resonates most with onboarding or reengagement, playing a subtle but powerful role in guiding behavior.

- Across all segments, the **average time to transaction is 6.3 days** , but certain groups, like Frequent Shoppers and Sleeping Buyers, take significantly longer. This insight reveals not just how customers behave, but when  giving a clearer picture of where patience is a virtue and where urgency is key.
#  Offer Journey
[Back to Table of Contents](#table-of-contents)
- **Frequent Shoppers** emerged as the most targeted group, having received an impressive **3,936 offers**. Yet despite leading in total completions (**448**), their **conversion rate sits at a modest ~11%**, suggesting that volume doesn’t always translate to action.

- In contrast, **First-Time Buyers** made a powerful impression. With **521 completions** from **2,612 offers**, they delivered a **standout 19.9% conversion rate** the highest of any segment. Clearly, these newcomers are highly responsive and receptive when engaged early.

- **High Spenders** and **VIP Customers**, while not leading in sheer numbers, showed **modest engagement**  likely a reflection of being more selective or receiving fewer, more curated offers.
- **Loyal Core** showed steady completion patterns, with **223** completions out of **1,488 offers**. This segment values consistency and benefits from ongoing engagement.**Potential Upgraders** received **276 offers** and completed **51**, reflecting a healthy **18.5% conversion rate**, indicating they are open to compelling value propositions.

- When it comes to **offer viewing**, engagement drops sharply. Only **12%** of all received offers were ever viewed  **1,393 views from 11,482 offers**. But again, **First-Time Buyers** rose above, showing the best **view-to-complete conversion**, confirming that **when they do open offers, they act**.

- On the flip side, **Win-Back Targets** barely registered. With just **2 views and 5 completions**, this segment appears almost entirely disengaged, signaling an urgent need for a new, more aggressive approach to reignite interest.

- Looking deeper into **drop-off patterns**, the biggest loss occurs between **offer received and offer viewed**, the first major friction point. This bottleneck highlights a simple truth: if customers don’t **see** the offer, they can’t act on it. Improving **visibility**, perhaps through better timing or standout messaging, could make a significant difference.

**Business Takeaway**

- **Promotional offers** (especially those requiring completion) are effective at driving revenue.
    
- **Informational campaigns**, while limited in direct revenue, may still influence awareness.
    
- **Improving attribution tracking** could uncover more insights into customer motivation.


# Turning Insight into Action from Final Takeaways

[Back to Table of Contents](#table-of-contents)

The data highlights a trio of highly efficient segments: **First-Time Buyers**, **High Spenders**, and **Potential Upgraders**. These groups consistently turn even limited exposure , whether just receiving or viewing offers  into completions. Their behavior makes them **ideal for ROI-focused campaigns**, where results matter most.

In contrast, **Frequent Shoppers** and **Sleeping Buyers**, despite being exposed to a high volume of offers, fall short in follow-through. To move the needle with these segments, campaigns may need to lean more into **urgency**, **personalized relevance**, or **tighter messaging** that sparks quicker action.

**Win-Back Targets** remain the most challenging group. Their low visibility and poor engagement suggest the need for a **fresh approach** , one that could include **reactivation offers**, **targeted retargeting**, or even reintroducing the brand’s value proposition from the ground up.

**Loyal Core**, with steady and reliable engagement, represents an opportunity to reinforce loyalty and expand their share of wallet. They **thrive on recognition and rewards**, so doubling down on consistent loyalty messaging, exclusive member perks, and tiered benefits will encourage continued engagement and gradual spend increases. These customers may not respond to heavy urgency cues, but they **value consistency, trust, and being acknowledged** for their ongoing relationship


For slower-moving clusters like **Frequent Shoppers** and **Sleeping Buyers**, consider integrating **urgency cues** or **shorter offer windows** to nudge quicker decisions. Meanwhile, **quick responders** like **VIPs** and **Win-Backs** can be driven with **flash deals** or **exclusive perks**, optimizing for speed and value.

Lastly, remember that **timing isn’t one-size-fits-all**. Segment-specific behaviors show that some customers need more time, while others move fast. **Tailoring the timing** of your campaigns accordingly could be the edge that boosts overall performance.
#  Final Recommendations
[Back to Table of Contents](#table-of-contents)
- **Cluster 1 – Frequent Shoppers (33.2% of Offer Attributions)**  
  - This group is highly engaged but slow to act (**13.29 days on average**). To drive faster conversions, use **offers with lower difficulty and shorter expiration windows** to create urgency. Since they are already active, reinforcing their behavior with **timely nudges** can increase offer completion rates. Consider mobile and in-app channels for real-time prompts.

- **Cluster 2 – First-Time Buyers (Highest Conversion Rate: 19.9%)**  
  - This segment responds quickly (**8.5 days**) and converts efficiently. Leverage their momentum with **welcome incentives**, **low-to-medium difficulty offers**, and **short durations** to reinforce early engagement. This group is ideal for **onboarding journeys** and **targeted welcome campaigns**, especially via mobile or email.

- **Cluster 3 – Sleeping Buyers (Delayed Action, Moderate Volume)**  
  - These reactivating users show potential but act slowly (**11.5 days**) and may need more time to decide. Use **medium-difficulty offers with longer durations**, paired with **awareness messaging** (e.g., informational offers) to rebuild engagement. Retargeting via **email or social** could reawaken interest.

- **Cluster 4 – High Spenders & VIP Customers (Strategic, Fast-Acting Segment)**  
  - Though smaller in size, these customers act **faster than average** and likely contribute high transaction value. Use **high-difficulty offers** and **medium durations** to maintain their attention while offering a challenge. **Exclusive access**, **early-bird deals**, or **loyalty tiers** via web or email can reinforce brand loyalty.

- **Cluster 5 – Win-Back Targets (Very Low Engagement, Fast When Active)**  
  - This disengaged group shows little offer interaction but acts quickly when they do respond. Use **flash deals**, **personalized reactivation messages**, or **exclusive offers** with **short durations**. These should be delivered through **email retargeting** or **SMS** to grab attention and drive urgency.
-  **Cluster 6 – Loyal Core (Consistent, Reliable engagement)**
   - Use **medium-difficulty, medium-duration offers** to maintain their high visit frequency and steady engagement. Consider **loyalty point accelerators** or **membership rewards**.Or  introduce **tiered loyalty programs** that reward consistent purchasing with points multipliers, exclusive member events, and early access to new products. **Personalized messaging** emphasizing their status as valued, long-term customers will strengthen brand attachment.

- **Cluster 7 – Potential Upgraders (Respond relatively quickly and show signs of growing interest)**   
  - Provide **higher-difficulty, longer-duration offers** to encourage increased spend and deepen loyalty.**Deploy progressive challenges** or **spend-based milestones** that unlock escalating rewards. Emphasize personalized product recommendations and curated bundles that make them feel special and incentivize larger purchases. Combine this with messaging that highlights how close they are to higher loyalty tiers or exclusive benefits.