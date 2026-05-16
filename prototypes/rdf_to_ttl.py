from rdflib import Graph

# g = Graph()
# g.parse("/Users/kelvinjou/Downloads/food.rdf", format="xml")
# g.serialize(destination="outputs/food.ttl", format="turtle")


# SELECT ?person means from all matching data return only value bounded to variable ?person
# "a" is a turtle shorthand for rdf:type
# ?person a ex:Person means ?person is of type Person
# ; Keep the same subject, and add another predicate/object pair.
""" QUERY """
# g = Graph()
# g.parse("outputs/ontology.ttl")

# query_str = """
# PREFIX ex: <http://example.org/>

# SELECT ?person
# WHERE {
#     ?person a ex:Person ;
#         ex:hasName "Alice" .
# }
# """

# for row in g.query(query_str):
#     print(row.person)


# SELECT x, x is a variable you invent inside the query
"""XR ontology query"""
g = Graph()
g.parse("outputs/xr-design.ttl")

query_str = """
PREFIX xr: <http://xrdesign.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?experienceLabel ?mediumLabel ?deviceLabel ?interactionLabel ?componentLabel
WHERE {
    ?experience a xr:XRExperience ;
                rdfs:label ?experienceLabel ;
                xr:usesMedium ?medium ;
                xr:runOnDevice ?device ;
                xr:employsInteraction ?interaction ;
                xr:containsComponent ?component .

    ?medium rdfs:label ?mediumLabel .
    ?device rdfs:label ?deviceLabel .
    ?interaction rdfs:label ?interactionLabel .
    ?component rdfs:label ?componentLabel .
}
ORDER BY ?experienceLabel
"""
for row in g.query(query_str):
    print(f"{row.experienceLabel} | {row.mediumLabel} | {row.deviceLabel} | {row.interactionLabel} | {row.componentLabel}")

"""
graph traversal and scoring (determining priorities)
determine which branch to explore
return design recommendations

given original query + chat history
add on references w/ RLM extracting relevant considerations of graph

give RLM access to top level classes (line 23 of ttl)
determine priorities, by assigning scores per connection?

ttl structure IS multihop reasoning
do I want to generate a XR experience TTL from prompt or...?

ENHANCED_XR HAS RELATIONAL PROPERTIES (i.e. supportsTask, appliesTo, addressesHumanFactor)

can it detect + resolve contradiction if RLMs run in parallel and come back?
"""
