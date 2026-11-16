"""Publish actual token-count summaries and complete per-request CSV attachments."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
from html import escape
import xml.etree.ElementTree as ET

from full_common import atomic_json
from priority_plan import key, versions
from request_statistics import summarize_requests
from sync_feishu import lark

HEADERS=["数据集","输出上限","并发","成功请求数","吞吐(tokens/s)","实际输出tokens：均值 / P50 / P95 / 范围","长度结束数","截断率"]
LABELS={"math500":"MATH-500","aime25":"AIME25","humaneval":"HumanEval","mbpp":"MBPP","mt-bench":"MT-Bench","alpaca":"Alpaca"}


def matrix(rows):
    output=[HEADERS]
    for row in sorted(rows,key=lambda r:(r["output_length"],r["concurrency"],list(LABELS).index(r["dataset"]))):
        s=row["output_statistics"]
        if s["unknown_finish_reason_requests"] or s["truncation_rate_percent"] is None:
            raise ValueError("Cannot publish an exact truncation rate without all finish reasons")
        output.append([LABELS[row["dataset"]],str(row["output_length"]),str(row["concurrency"]),
            str(s["successful_requests"]),f"{row['output_tokens_per_second']:.3f}",
            f"{s['actual_output_tokens_mean']:.3f} / {s['actual_output_tokens_p50']:.3f} / {s['actual_output_tokens_p95']:.3f} / {s['actual_output_tokens_min']}–{s['actual_output_tokens_max']}",
            str(s["length_ended_requests"]),f"{s['truncation_rate_percent']:.2f}%"])
    return output


def table_xml(values):
    widths=[100,90,60,100,110,270,100,100]
    columns="<colgroup>"+"".join(f'<col width="{w}"/>' for w in widths)+"</colgroup>"
    body="".join("<tr>"+"".join("<td><p>"+escape(str(c))+"</p></td>" for c in row)+"</tr>" for row in values)
    return "<table>"+columns+"<tbody>"+body+"</tbody></table>"


def table_values(table):
    return [["".join(c.itertext()).strip() for c in row if c.tag in ("td","th")] for row in table.iter("tr")]


def find_table(root, marker):
    children=list(root)
    for i,node in enumerate(children):
        if node.tag=="p" and "".join(node.itertext()).strip()==marker:
            if i+1<len(children) and children[i+1].tag=="table":
                return children[i+1]
            raise RuntimeError("Statistics marker exists without its table")
    return None


def write_csv(args,unit,rows):
    unit_id=unit["unit_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+",unit_id):
        raise ValueError("Invalid unit ID")
    remote=f"/root/workspace/sglang/rebuttal/output/{args.phase}/measurements/{unit_id}/requests.jsonl"
    done=subprocess.run(["ssh","-o","ConnectTimeout=15","h20","cat "+shlex.quote(remote)],capture_output=True,text=True,check=True,timeout=90)
    requests={}
    for line in done.stdout.splitlines():
        item=json.loads(line)
        requests[item["case_id"],item["created_at"]]=item["requests"]
    content=io.StringIO(newline="")
    fields=["case_id","dataset","concurrency","output_limit","request_index","success","actual_output_tokens","finish_reason","response_id"]
    writer=csv.DictWriter(content,fieldnames=fields);writer.writeheader()
    count=0
    for row in sorted(rows,key=lambda r:key(r)):
        actual=requests[row["case_id"],row["created_at"]]
        if summarize_requests(actual)!=row["output_statistics"]:
            raise RuntimeError("Per-request records differ from their published summary")
        if len(actual)!=row["total_requests"]:
            raise RuntimeError("Per-request file is incomplete")
        for i,r in enumerate(actual):
            writer.writerow(dict(case_id=row["case_id"],dataset=row["dataset"],concurrency=row["concurrency"],
                output_limit=row["output_length"],request_index=r.get("request_index",i),success=r["success"],
                actual_output_tokens=r["completion_tokens"],finish_reason=r.get("finish_reason"),response_id=r.get("response_id")))
            count+=1
    data=content.getvalue().encode("utf-8-sig")
    digest=hashlib.sha256(data).hexdigest()
    folder=args.output/"request-details";folder.mkdir(exist_ok=True)
    path=folder/(unit_id+"-"+digest[:12]+".csv");path.write_bytes(data)
    return path,count,digest


def publish(args,payload,state,fetch,fetch_table,update,doc):
    root=fetch()
    heading_marker="REQUEST_STATISTICS_SCOPE_v3"
    if args.phase=="dflash4l_v1":
        heading_marker="DFLASH_4L_REQUEST_STATISTICS_v1"
        if heading_marker not in "".join(root.itertext()):
            update("append","<h1>DFlash-4L 请求实际输出与截断率</h1><p>DFLASH_4L_REQUEST_STATISTICS_v1："
                "目标Qwen3-4B，草稿dflash-4L/step_2616，共4层；六个数据集全部行，并发1/2/4/8/16，固定输出512，Graph256，共30项。"
                "实际长度读取usage.completion_tokens；截断率=成功请求中finish_reason为length的请求数÷全部成功请求数×100%。"
                "保持ignore_eos=true的固定输出口径，实际结束原因逐请求保存；完整配置结束后附7650条请求CSV。</p>")
        root=fetch()
    if heading_marker not in "".join(root.itertext()):
        update("append","<h1>请求实际输出长度与截断率</h1><p>REQUEST_STATISTICS_SCOPE_v3：仅执行顺序1、2，共120个组合。"
            "每行统计一个目标模型/草稿模型/数据集/并发/输出上限组合；实际长度读取usage.completion_tokens，"
            "截断率=成功请求中finish_reason为length的请求数÷全部成功生成请求数×100%。结束原因缺失时不填截断率。"
            "长度同时给出均值、P50、P95、最小值与最大值；百分位数使用线性插值。"
            "当前固定输出设置ignore_eos=true，因此达到上限的比例通常为100%，以真实结束原因为准。"
            "每种目标/草稿配置测完后附上逐请求CSV，并在吞吐及输出统计回读核验后进入下一配置。"
            "此前未保存结束原因的18项已保留原始结果，本轮补测以取得完整统计。</p>")
    stats_state=state.setdefault("request_statistics",{})
    ready=set()
    for unit in payload.get("units",[]):
        rows=[r for r in payload["rows"] if key(r) in unit["versions"]]
        if not rows:
            continue
        values=matrix(rows)
        digest=hashlib.sha256(json.dumps(values,ensure_ascii=False).encode()).hexdigest()
        previous=stats_state.get(unit["unit_id"],{})
        marker="REQUEST_STATISTICS_v3 "+unit["unit_id"]
        if previous.get("sha256")!=digest or previous.get("versions")!=unit["versions"]:
            table=find_table(fetch(),marker)
            if table is None:
                update("append","<h2>"+escape(unit["family"]+" / "+unit["variant"])+"</h2><p>"+marker+"</p>"+table_xml(values))
            else:
                actual=table_values(table)
                if actual!=values:
                    if previous and actual!=previous.get("values"):
                        raise RuntimeError("Statistics table was edited externally")
                    update("block_replace",table_xml(values),table.attrib["id"])
            table=find_table(fetch(),marker)
            if table is None or table_values(table)!=values:
                raise RuntimeError("Statistics table readback differs")
            same_versions=previous.get("versions")==unit["versions"]
            stats_state[unit["unit_id"]]=dict(sha256=digest,versions=unit["versions"],values=values,
                csv_sha256=previous.get("csv_sha256") if same_versions else None,
                csv_name=previous.get("csv_name") if same_versions else None)
            atomic_json(args.output/"template-publication-state.json",state)
            print("STATISTICS_VERIFIED "+unit["unit_id"]+" cells="+str(len(rows)),flush=True)
        if unit["complete"]:
            saved=stats_state[unit["unit_id"]]
            if not saved.get("csv_sha256"):
                path,count,csv_hash=write_csv(args,unit,rows)
                full=ET.tostring(fetch(),encoding="unicode")
                if path.name not in full:
                    relative=path.resolve().relative_to(Path.cwd())
                    lark("+media-insert","--doc",doc,"--file",str(relative),"--type","file")
                full=ET.tostring(fetch(),encoding="unicode")
                if path.name not in full:
                    raise RuntimeError("Per-request CSV attachment was not found on readback")
                saved.update(csv_sha256=csv_hash,csv_name=path.name,request_count=count)
                atomic_json(args.output/"template-publication-state.json",state)
                print("REQUEST_CSV_VERIFIED "+unit["unit_id"]+" requests="+str(count),flush=True)
            ready.add(unit["unit_id"])
    return ready
