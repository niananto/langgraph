"""
schema_stress_test.py — how well does llama3.1:8b populate tools with
increasingly nested argument schemas?

Four tools, four degrees of schema complexity:

  D1 — all flat primitives           (book_restaurant)
  D2 — top-level objects             (plan_vacation)
  D3 — objects inside objects        (place_order)
  D4 — objects inside objects inside objects inside objects (deploy_application)

Each tool is tested with a single crafted prompt that requires the model to
fill every field correctly.  The script prints:

  [SCHEMA]  — the JSON schema LangChain derived from the function signature,
               exactly what gets serialised into the prompt
  [RAW]     — AIMessage.content (raw token text)
  [PARSED]  — AIMessage.tool_calls after langchain_ollama normalises them
  [RESULT]  — what the (fake) tool returned

Run:
    python schema_stress_test.py
    python schema_stress_test.py --model llama3.2:3b
"""

from __future__ import annotations

import json
import sys
from typing import Annotated

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL = next(
    (sys.argv[sys.argv.index("--model") + 1] for _ in ["x"] if "--model" in sys.argv),
    "llama3.1:8b",
)

SEP  = "=" * 70
SEP2 = "-" * 70


# ---------------------------------------------------------------------------
# DEGREE 1 — all flat primitives
# ---------------------------------------------------------------------------
def book_restaurant(
    restaurant_name: str,
    date: str,
    time: str,
    party_size: int,
    outdoor_seating: bool,
) -> str:
    """Book a table at a restaurant.

    Args:
        restaurant_name: Name of the restaurant.
        date: Reservation date in YYYY-MM-DD format.
        time: Reservation time in HH:MM (24h) format.
        party_size: Number of guests.
        outdoor_seating: Whether outdoor seating is preferred.
    """
    return json.dumps({
        "confirmation_id": "RSV-001",
        "restaurant_name": restaurant_name,
        "date": date,
        "time": time,
        "party_size": party_size,
        "outdoor_seating": outdoor_seating,
        "status": "confirmed",
    })


# ---------------------------------------------------------------------------
# DEGREE 2 — top-level objects
# ---------------------------------------------------------------------------
class Traveler(BaseModel):
    name: str = Field(description="Full name of the traveller.")
    age: int = Field(description="Age in years.")
    passport_number: str = Field(description="Passport number.")

class Budget(BaseModel):
    amount: int = Field(description="Numerical budget value.")
    currency: str = Field(description="ISO 4217 currency code, e.g. USD.")

class Preferences(BaseModel):
    accommodation_type: str = Field(description="E.g. hotel, hostel, airbnb.")
    activity_focus: str = Field(description="E.g. adventure, culture, relaxation.")


def plan_vacation(
    destination: str,
    duration_days: int,
    traveler: Traveler,
    budget: Budget,
    preferences: Preferences,
) -> str:
    """Plan a vacation itinerary.

    Args:
        destination: City or country to travel to.
        duration_days: Length of the trip in days.
        traveler: Information about the traveller.
        budget: Total travel budget.
        preferences: Accommodation and activity preferences.
    """
    return json.dumps({
        "plan_id": "PLN-002",
        "destination": destination,
        "duration_days": duration_days,
        "traveler": traveler.model_dump(),
        "budget": budget.model_dump(),
        "preferences": preferences.model_dump(),
        "status": "itinerary_generated",
    })


# ---------------------------------------------------------------------------
# DEGREE 3 — objects inside objects
# ---------------------------------------------------------------------------
class Address(BaseModel):
    street: str = Field(description="Street address including house/apt number.")
    city: str = Field(description="City name.")
    country: str = Field(description="Country name.")
    zip_code: str = Field(description="Postal / ZIP code.")

class ShippingDetails(BaseModel):
    address: Address = Field(description="Delivery address.")
    method: str = Field(description="Shipping method: standard or express.")
    leave_at_door: bool = Field(description="Whether to leave the parcel at the door.")

class CardDetails(BaseModel):
    number: str = Field(description="16-digit card number.")
    expiry: str = Field(description="Card expiry in MM/YY format.")
    cvv: str = Field(description="3 or 4 digit security code.")

class PaymentDetails(BaseModel):
    method: str = Field(description="Payment method: card or paypal.")
    card: CardDetails = Field(description="Card details (required when method is card).")


def place_order(
    product_id: str,
    quantity: int,
    shipping: ShippingDetails,
    payment: PaymentDetails,
) -> str:
    """Place a product order with shipping and payment details.

    Args:
        product_id: Unique identifier of the product.
        quantity: Number of units to order.
        shipping: Shipping address and delivery preferences.
        payment: Payment method and card details.
    """
    return json.dumps({
        "order_id": "ORD-003",
        "product_id": product_id,
        "quantity": quantity,
        "shipping": shipping.model_dump(),
        "payment": {"method": payment.method},  # don't echo card details
        "status": "order_placed",
    })


# ---------------------------------------------------------------------------
# DEGREE 4 — objects inside objects inside objects inside objects
# ---------------------------------------------------------------------------
class ScalingPolicy(BaseModel):
    metric: str = Field(description="Scaling metric: cpu_utilization or memory_utilization.")
    target_threshold_pct: int = Field(description="Target utilisation percentage to trigger scaling.")
    cooldown_seconds: int = Field(description="Seconds to wait between scaling events.")

class Autoscaling(BaseModel):
    enabled: bool = Field(description="Whether autoscaling is active.")
    min_nodes: int = Field(description="Minimum number of nodes.")
    max_nodes: int = Field(description="Maximum number of nodes.")
    policy: ScalingPolicy = Field(description="Scaling trigger policy.")

class NodePool(BaseModel):
    size: int = Field(description="Initial number of nodes.")
    machine_type: str = Field(description="VM machine type, e.g. n2-standard-4.")
    autoscaling: Autoscaling = Field(description="Autoscaling configuration for this pool.")

class Cluster(BaseModel):
    name: str = Field(description="Cluster name.")
    region: str = Field(description="Cloud region, e.g. us-central1.")
    node_pool: NodePool = Field(description="Primary node pool configuration.")

class ResourceQuota(BaseModel):
    request: str = Field(description="Guaranteed resource, e.g. '250m' for CPU or '512Mi' for memory.")
    limit: str = Field(description="Maximum resource allowed, e.g. '1' for CPU or '1Gi' for memory.")

class AppResources(BaseModel):
    cpu: ResourceQuota = Field(description="CPU request and limit.")
    memory: ResourceQuota = Field(description="Memory request and limit.")


def deploy_application(
    app_name: str,
    image_tag: str,
    cluster: Cluster,
    resources: AppResources,
) -> str:
    """Deploy a containerised application to a cloud cluster.

    Args:
        app_name: Name of the application to deploy.
        image_tag: Docker image tag to deploy, e.g. v1.2.3.
        cluster: Target cluster and node pool configuration.
        resources: CPU and memory quotas for the application container.
    """
    return json.dumps({
        "deployment_id": "DEP-004",
        "app_name": app_name,
        "image_tag": image_tag,
        "cluster": cluster.model_dump(),
        "resources": resources.model_dump(),
        "status": "deploying",
    })


# ---------------------------------------------------------------------------
# Test cases — one prompt per degree
# ---------------------------------------------------------------------------
TESTS = [
    {
        "degree": 1,
        "tool": book_restaurant,
        "prompt": (
            "Book a table for 4 people at 'The Golden Fork' restaurant "
            "on 2026-07-15 at 19:30. We'd prefer outdoor seating."
        ),
    },
    {
        "degree": 2,
        "tool": plan_vacation,
        "prompt": (
            "Plan a 10-day vacation to Tokyo for traveller Alice Smith, "
            "age 29, passport number AB123456. "
            "Total budget is 3000 USD. "
            "She prefers a hotel and wants a culture-focused trip."
        ),
    },
    {
        "degree": 3,
        "tool": place_order,
        "prompt": (
            "Order 2 units of product PROD-789. "
            "Ship it to 42 Maple Street, Springfield, USA, ZIP 62701 via express shipping — "
            "leave it at the door. "
            "Pay by card: number 4111111111111111, expiry 09/27, CVV 321."
        ),
    },
    {
        "degree": 4,
        "tool": deploy_application,
        "prompt": (
            "Deploy the app 'payments-service' using image tag v2.5.1. "
            "Target cluster: 'prod-cluster' in us-central1 region. "
            "Node pool: 3 initial nodes of type n2-standard-4 with autoscaling enabled "
            "between 2 and 10 nodes, scaling on cpu_utilization at 70% threshold "
            "with a 120 second cooldown. "
            "Container resources: CPU request 250m / limit 1, memory request 512Mi / limit 1Gi."
        ),
    },
]


# ---------------------------------------------------------------------------
# Graph builder — one fresh graph per test (one tool at a time)
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


def run_test(degree: int, tool, prompt: str) -> None:
    print(f"\n{SEP}")
    print(f"DEGREE {degree}  —  tool: {tool.__name__}")
    print(SEP)

    # print the schema the model will see
    schema = convert_to_openai_tool(tool)
    print("\n[SCHEMA]")
    print(json.dumps(schema, indent=2))

    llm = ChatOllama(model=MODEL, temperature=0).bind_tools([tool])

    def call_model(state: State) -> State:
        response = llm.invoke(state["messages"])
        print(f"\n{SEP2}")
        print(f"[RAW]    content = {repr(response.content)}")
        if response.tool_calls:
            print("[PARSED] tool_calls =")
            for tc in response.tool_calls:
                print(f"           name={tc['name']!r}")
                print(f"           args={json.dumps(tc['args'], indent=11)}")
        else:
            print("[PARSED] tool_calls = (none — model did not call the tool)")
        print(SEP2)
        return {"messages": [response]}

    _tool_node = ToolNode([tool])

    def run_tools(state: State) -> State:
        result = _tool_node.invoke(state)
        for msg in result["messages"]:
            print(f"[RESULT] {msg.content}")
        return result

    graph = (
        StateGraph(State)
        .add_node("model", call_model)
        .add_node("tools", run_tools)
        .add_edge(START, "model")
        .add_conditional_edges("model", tools_condition)
        .add_edge("tools", "model")
        .compile()
    )

    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Call the appropriate tool with all required arguments.",
    }
    graph.invoke({"messages": [system_msg, {"role": "user", "content": prompt}]})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Model: {MODEL}")
    print("Testing tool schema complexity: D1 (flat) → D4 (deeply nested)\n")

    for test in TESTS:
        run_test(**test)

    print(f"\n{SEP}")
    print("All tests complete.")
    print(SEP)


if __name__ == "__main__":
    main()
